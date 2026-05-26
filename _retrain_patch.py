        EPOCHS = 3
        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            metrics = trainer.train_epoch(
                str(MASTER_CORPUS_PATH),
                batch_size=32,
                on_batch_complete=None
            )
            elapsed = round(time.time() - t0, 2)
            print(f"  Epoch {epoch}/{EPOCHS}  records={metrics['records']}  "
                  f"loss={metrics['loss']:.5f}  mse={metrics['mse']:.5f}  "
                  f"ce={metrics['ce']:.5f}  {elapsed}s")
