本仓库是对_VF.lstm()函数的Python版本实现。

如果阅读PyTorch中LSTM实现源码可以发现，LSTM最底层是通过_VF.lstm()函数实现的，这个函数在C++支持库中，参考https://github.com/pytorch/pytorch/blob/49777e67303f608987ec0948c7fd8f46f6d3ca83/torch/csrc/api/src/nn/modules/rnn.cpp。

底层C语言可以保证LSTM的运行效率，但当需要对LSTM结构进行一些修改时却很不方便，因此本文基于python语言实现了一版，customrnns包中是我实现出来的LSTM结构，目前只实现了LSTM，并没有RNN和GRU版本的实现，不过应该也类似。

注意：当前版本暂不支持设置LSTM中的 bidirectional、proj_size、dropout 参数。

参考资料：
（fastrnns包中的代码文件）
https://github.com/pytorch/pytorch/blob/e9ef087d2d12051341db485c8ac64ea64649823d/benchmarks/fastrnns/cells.py
https://github.com/pytorch/pytorch/blob/e9ef087d2d12051341db485c8ac64ea64649823d/benchmarks/fastrnns/custom_lstms.py