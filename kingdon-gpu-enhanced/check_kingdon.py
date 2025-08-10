try:
    import kingdon
    print(f'Kingdon location: {kingdon.__file__}')
    print(f'Kingdon version: {getattr(kingdon, "__version__", "unknown")}')
    print('Kingdon is currently installed')
except ImportError:
    print('Kingdon not installed')