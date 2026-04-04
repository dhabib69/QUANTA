from QUANTA_bot import Bot

try:
    print("Initializing Bot...")
    bot = Bot()
    print("Bot initialized. Starting training...")
    bot._train_models(clean_retrain=True)
    print("Training finished without exception.")
except Exception as e:
    import traceback
    traceback.print_exc()
