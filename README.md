# gradient-checkpointing


To run tests:
```
cd test
export TF_CPP_MIN_LOG_LEVEL=1
./run_all_tests.sh
python resnet_test.py
python pixel_cnn_test.py
```

If not assertion failures, tests pass
