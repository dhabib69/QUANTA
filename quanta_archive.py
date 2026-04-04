"""
QUANTA Archive Downloader - Binance Vision
Bypasses slow rate-limited REST API to download monthly ZIP files directly from data.binance.vision
"""
import os
import io
import time
import zipfile
import asyncio
import aiohttp
import logging
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

class BinanceArchiveDownloader:
    def __init__(self, cache_instance):
        self.base_url = "https://data.binance.vision/data/futures/um/monthly/klines"
        self.cache = cache_instance
        self.max_concurrent_downloads = 10
        self.columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
            'taker_buy_quote', 'ignore'
        ]

    def _generate_month_urls(self, symbol, interval, total_days):
        """Generate a list of URLs and month strings needed to cover the requested days."""
        urls = []
        months = []
        
        # We start from previous month (current month is usually incomplete/daily only)
        # Using 1st day of current month to ensure we get full past months
        current_date = datetime.now()
        end_date = current_date.replace(day=1) - relativedelta(days=1)
        start_date = current_date - relativedelta(days=total_days)
        
        # Iterate month by month from start to end
        # We go from end_date backwards to reach start_date
        iter_date = end_date
        while iter_date > start_date.replace(day=1):
            yyyy_mm = iter_date.strftime("%Y-%m")
            # Format: https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-2024-01.zip
            file_name = f"{symbol}-{interval}-{yyyy_mm}.zip"
            url = f"{self.base_url}/{symbol}/{interval}/{file_name}"
            
            urls.append(url)
            months.append(yyyy_mm)
            
            # Move back 1 month
            iter_date = iter_date.replace(day=1) - relativedelta(days=1)
            
        return urls, months

    async def _download_and_extract_month(self, session, url, symbol, yyyy_mm, proxy=None):
        """Download one month's ZIP and extract the CSV into a pandas DataFrame."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Explicitly pass proxy if available
                async with session.get(url, proxy=proxy) as response:
                    if response.status == 404:
                        return None  # Missing month (e.g. before coin was listed)

                    if response.status != 200:
                        logging.warning(f"Failed to download {url}: HTTP {response.status}")
                        return None

                    # Read zip file bytes into memory
                    zip_data = await response.read()

                    # Extract CSV
                    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
                        csv_filename = z.namelist()[0]
                        with z.open(csv_filename) as csv_file:
                            df = pd.read_csv(csv_file, header=None, low_memory=False)

                            if str(df.iloc[0, 0]) == 'open_time':
                                df = df.iloc[1:].copy()

                            df.columns = self.columns
                            df['open_time'] = pd.to_numeric(df['open_time'])
                            df['close_time'] = pd.to_numeric(df['close_time'])
                            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base', 'taker_buy_quote']:
                                df[col] = pd.to_numeric(df[col])
                            df['trades'] = pd.to_numeric(df['trades']).astype('int64')

                            return df

            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logging.warning(f"Retry {attempt+1}/{max_retries} for {symbol} {yyyy_mm}: {e} — waiting {wait}s")
                    await asyncio.sleep(wait)
                else:
                    logging.error(f"Error downloading {yyyy_mm} for {symbol}: {e}")
        return None

    async def _download_all_months(self, symbol, interval, urls, months):
        """Manage concurrent downloads using a semaphore."""
        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        
        # Check if proxy is set in environment or config
        proxy = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
        
        async def bounded_download(session, url, yyyy_mm):
            async with semaphore:
                await asyncio.sleep(0.1)
                return await self._download_and_extract_month(session, url, symbol, yyyy_mm, proxy)
                
        connector = aiohttp.TCPConnector(limit=self.max_concurrent_downloads)
        # 10 minute timeout per zip to handle large network spikes
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=True) as session:
            tasks = []
            for url, yyyy_mm in zip(urls, months):
                tasks.append(bounded_download(session, url, yyyy_mm))
                
            results = await asyncio.gather(*tasks)
            return results

    def fetch_historical_archive(self, symbol, interval, days):
        """
        Main entrypoint. Downloads up to 'days' of history from Binance Vision,
        merges it, and saves it directly to FeatherCache.
        """
        urls, months = self._generate_month_urls(symbol, interval, days)
        if not urls:
            return None
            
        print(f"   [FETCH] Fetching {len(urls)} monthly ZIP archives from Binance Vision...")
        t0 = time.time()
        
        try:
            # Run async loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        dataframes = loop.run_until_complete(self._download_all_months(symbol, interval, urls, months))
        
        # Filter out None (failed/404)
        valid_dfs = [df for df in dataframes if df is not None and not df.empty]
        
        if not valid_dfs:
            print(f"   [MISSING] No archive data available for {symbol}")
            return None
            
        # Merge all months
        print(f"   [MERGE] Merging {len(valid_dfs)} months of data...")
        master_df = pd.concat(valid_dfs, ignore_index=True)
        master_df.sort_values('open_time', inplace=True)
        master_df.drop_duplicates(subset=['open_time'], keep='last', inplace=True)
        
        # Save to Feather Cache directly
        try:
            # We convert DF to list format for cache.set
            # FeatherCache converts it right back to DF, but this ensures standard structure
            klines_list = master_df.values.tolist()
            if self.cache:
                self.cache.set(symbol, interval, klines_list)
            
            elapsed = time.time() - t0
            print(f"   [DONE] Fetched {len(klines_list):,} candles via ZIP Archive in {elapsed:.1f}s")
            return klines_list
            
        except Exception as e:
            logging.error(f"Failed to save archive data to cache for {symbol}: {e}")
            return None
