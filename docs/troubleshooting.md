# 🔧 Troubleshooting Guide

## Common Issues

### Installation Problems

**Issue**: `torch not found`
**Solution**: Install PyTorch first
```bash
pip install torch torchvision
```

**Issue**: `scipy missing`
**Solution**: Install scientific computing stack
```bash
pip install scipy numpy matplotlib
```

### Runtime Errors

**Issue**: `No temporal signals extracted`
**Solution**: Check that model has Conv2d or Linear layers with hooks

**Issue**: `FFT analysis fails`
**Solution**: Ensure sufficient signal length (>= 10 time points)

**Issue**: `CIFAR-10 download fails`
**Solution**: Check internet connection or use local data

### Performance Issues

**Issue**: Analysis takes too long
**Solution**: Reduce `max_batches` parameter

**Issue**: Memory errors
**Solution**: Reduce batch size or use smaller models

### Results Issues

**Issue**: All frequencies are zero
**Solution**: Check model is in correct mode (eval vs train)

**Issue**: State classification always returns 'idle'
**Solution**: Verify input data is not all zeros

**Issue**: Cross-modal validation fails
**Solution**: Ensure both NN-EEG and NN-fMRI results are properly formatted

## Getting Help

1. Check this troubleshooting guide
2. Review `docs/getting-started.md` for basic usage
3. Examine working examples in `examples/`
4. Run the test suite: `pytest tests/`
5. Check GitHub issues for similar problems

## Reporting Bugs

Include in your bug report:
- Python version
- PyTorch version  
- Complete error message
- Minimal code to reproduce
- System information (OS, hardware)

## FAQ

**Q: Can I use my own model architecture?**
A: Yes, framework works with any PyTorch model

**Q: How accurate are the results?**
A: Both NN-EEG and NN-fMRI have been validated on multiple datasets including CIFAR-10.

**Q: Can I use this in production?**
A: Yes, the complete dual-modal framework is production-ready and tested.
