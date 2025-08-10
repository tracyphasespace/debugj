try:
    import cupy
    print('CuPy version:', cupy.__version__)
    print('CUDA available:', cupy.cuda.is_available())
    print('GPU count:', cupy.cuda.runtime.getDeviceCount())
except ImportError:
    print('CuPy not installed')
except Exception as e:
    print('CuPy error:', str(e))