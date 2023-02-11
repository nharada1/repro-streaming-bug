# Sample command

```sh
$ pip install -r requirements.txt
$ main.py
```

# Sample output
```sh
Traceback (most recent call last):
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/trainer/call.py", line 38, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 650, in _fit_impl
    self._run(model, ckpt_path=self.ckpt_path)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 1112, in _run
    results = self._run_stage()
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 1191, in _run_stage
    self._run_train()
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 1214, in _run_train
    self.fit_loop.run()
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/loops/loop.py", line 199, in run
    self.advance(*args, **kwargs)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/loops/fit_loop.py", line 267, in advance
    self._outputs = self.epoch_loop.run(self._data_fetcher)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/loops/loop.py", line 199, in run
    self.advance(*args, **kwargs)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/loops/epoch/training_epoch_loop.py", line 187, in advance
    batch = next(data_fetcher)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/utilities/fetching.py", line 184, in __next__
    return self.fetching_function()
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/utilities/fetching.py", line 265, in fetching_function
    self._fetch_next_batch(self.dataloader_iter)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/utilities/fetching.py", line 280, in _fetch_next_batch
    batch = next(iterator)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/trainer/supporters.py", line 569, in __next__
    return self.request_next_batch(self.loader_iters)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/trainer/supporters.py", line 581, in request_next_batch
    return apply_to_collection(loader_iters, Iterator, next)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/lightning_utilities/core/apply_func.py", line 47, in apply_to_collection
    return function(data, *args, **kwargs)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 628, in __next__
    data = self._next_data()
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 671, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 34, in fetch
    data.append(next(self.dataset_iter))
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/streaming/base/dataset.py", line 727, in __iter__
    sample_ids = self._get_partition(world, epoch, sample_in_epoch)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/streaming/base/dataset.py", line 384, in _get_partition
    num_bytes = os.path.getsize(filename)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/genericpath.py", line 50, in getsize
    return os.stat(filename).st_size
FileNotFoundError: [Errno 2] No such file or directory: '/tmp/streaming/ef5956/shuffle.npy'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/nharada/Code/mosaic-streaming-crash-ddp/main.py", line 110, in <module>
    main()
  File "/home/nharada/Code/mosaic-streaming-crash-ddp/main.py", line 106, in main
    trainer.fit(mnist_model, train_loader)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 608, in fit
    call._call_and_handle_interrupt(
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/trainer/call.py", line 59, in _call_and_handle_interrupt
    trainer.strategy.reconciliate_processes(traceback.format_exc())
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/strategies/ddp.py", line 460, in reconciliate_processes
    raise DeadlockDetectedException(f"DeadLock detected from rank: {self.global_rank} \n {trace}")
pytorch_lightning.utilities.exceptions.DeadlockDetectedException: DeadLock detected from rank: 2 
 Traceback (most recent call last):
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/trainer/call.py", line 38, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 650, in _fit_impl
    self._run(model, ckpt_path=self.ckpt_path)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 1112, in _run
    results = self._run_stage()
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 1191, in _run_stage
    self._run_train()
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 1214, in _run_train
    self.fit_loop.run()
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/loops/loop.py", line 199, in run
    self.advance(*args, **kwargs)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/loops/fit_loop.py", line 267, in advance
    self._outputs = self.epoch_loop.run(self._data_fetcher)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/loops/loop.py", line 199, in run
    self.advance(*args, **kwargs)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/loops/epoch/training_epoch_loop.py", line 187, in advance
    batch = next(data_fetcher)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/utilities/fetching.py", line 184, in __next__
    return self.fetching_function()
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/utilities/fetching.py", line 265, in fetching_function
    self._fetch_next_batch(self.dataloader_iter)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/utilities/fetching.py", line 280, in _fetch_next_batch
    batch = next(iterator)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/trainer/supporters.py", line 569, in __next__
    return self.request_next_batch(self.loader_iters)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/pytorch_lightning/trainer/supporters.py", line 581, in request_next_batch
    return apply_to_collection(loader_iters, Iterator, next)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/lightning_utilities/core/apply_func.py", line 47, in apply_to_collection
    return function(data, *args, **kwargs)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 628, in __next__
    data = self._next_data()
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 671, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 34, in fetch
    data.append(next(self.dataset_iter))
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/streaming/base/dataset.py", line 727, in __iter__
    sample_ids = self._get_partition(world, epoch, sample_in_epoch)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/streaming/base/dataset.py", line 384, in _get_partition
    num_bytes = os.path.getsize(filename)
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/genericpath.py", line 50, in getsize
    return os.stat(filename).st_size
FileNotFoundError: [Errno 2] No such file or directory: '/tmp/streaming/ef5956/shuffle.npy'

/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/multiprocessing/resource_tracker.py:224: UserWarning: resource_tracker: There appear to be 3 leaked shared_memory objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/multiprocessing/resource_tracker.py:224: UserWarning: resource_tracker: There appear to be 3 leaked shared_memory objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/multiprocessing/resource_tracker.py:237: UserWarning: resource_tracker: '/ef5956_next_epoch': [Errno 2] No such file or directory: '/ef5956_next_epoch'
  warnings.warn('resource_tracker: %r: %s' % (name, e))
/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/multiprocessing/resource_tracker.py:237: UserWarning: resource_tracker: '/ef5956_barrier': [Errno 2] No such file or directory: '/ef5956_barrier'
  warnings.warn('resource_tracker: %r: %s' % (name, e))
/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/multiprocessing/resource_tracker.py:237: UserWarning: resource_tracker: '/ef5956_shard_states': [Errno 2] No such file or directory: '/ef5956_shard_states'
  warnings.warn('resource_tracker: %r: %s' % (name, e))
/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/multiprocessing/resource_tracker.py:224: UserWarning: resource_tracker: There appear to be 3 leaked shared_memory objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/multiprocessing/resource_tracker.py:237: UserWarning: resource_tracker: '/ef5956_next_epoch': [Errno 2] No such file or directory: '/ef5956_next_epoch'
  warnings.warn('resource_tracker: %r: %s' % (name, e))
/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/multiprocessing/resource_tracker.py:237: UserWarning: resource_tracker: '/ef5956_shard_states': [Errno 2] No such file or directory: '/ef5956_shard_states'
  warnings.warn('resource_tracker: %r: %s' % (name, e))
/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/multiprocessing/resource_tracker.py:237: UserWarning: resource_tracker: '/ef5956_barrier': [Errno 2] No such file or directory: '/ef5956_barrier'
  warnings.warn('resource_tracker: %r: %s' % (name, e))
Exception ignored in: <function StreamingDataset.__del__ at 0x7f7ecd171480>
Traceback (most recent call last):
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/streaming/base/dataset.py", line 814, in __del__
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/streaming/base/dataset.py", line 802, in _cleanup_shared_memory
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/multiprocessing/shared_memory.py", line 240, in unlink
ImportError: sys.meta_path is None, Python is likely shutting down
Exception ignored in: <function SharedBarrier.__del__ at 0x7f7ecd14f490>
Traceback (most recent call last):
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/site-packages/streaming/base/shared.py", line 72, in __del__
  File "/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/multiprocessing/shared_memory.py", line 240, in unlink
ImportError: sys.meta_path is None, Python is likely shutting down
/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/multiprocessing/resource_tracker.py:224: UserWarning: resource_tracker: There appear to be 3 leaked shared_memory objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/multiprocessing/resource_tracker.py:237: UserWarning: resource_tracker: '/ef5956_barrier': [Errno 2] No such file or directory: '/ef5956_barrier'
  warnings.warn('resource_tracker: %r: %s' % (name, e))
/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/multiprocessing/resource_tracker.py:237: UserWarning: resource_tracker: '/ef5956_shard_states': [Errno 2] No such file or directory: '/ef5956_shard_states'
  warnings.warn('resource_tracker: %r: %s' % (name, e))
/home/nharada/Software/mambaforge/envs/mosaic-streaming-crash-ddp/lib/python3.10/multiprocessing/resource_tracker.py:237: UserWarning: resource_tracker: '/ef5956_next_epoch': [Errno 2] No such file or directory: '/ef5956_next_epoch'
  warnings.warn('resource_tracker: %r: %s' % (name, e))
```