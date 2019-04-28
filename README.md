# rroi-align

RROI align layer used by vision team

## Build

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## Usage

```
# Generate a random testcase into a file named "testcase"
./gen_testcases
# Run rroi pooling.
# The golden result of top layer will be written into a file named "golden"
# The result of top layer using optimized CUDA kernel will be written into a file named "output"
./test_rroi
# Test result correctness
diff golden output > /dev/null
echo $?
```
