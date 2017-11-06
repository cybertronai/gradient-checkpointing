# gradient-checkpointing


To run tests:
```
cd test
pip install pytest
export TF_CPP_MIN_LOG_LEVEL=1
./run_all_tests.sh
pytest memory_test.py
pytest linearize_test.py
python pixel_cnn_test.py
python resnet_test.py
```

If not assertion failures, tests pass
