# nowcast_lstm_matlab

## *work in progress

MATLAB wrapper for [nowcast_lstm](https://github.com/dhopp1/nowcast_lstm) Python library. Long short-term memory neural networks for economic nowcasting. More background in [this](https://unctad.org/webflyer/economic-nowcasting-long-short-term-memory-artificial-neural-networks-lstm) UNCTAD research paper.

## Poll
To help me better understand the potential userbase and determine whether / which languages to develop future wrappers for, please answer the following poll on your programming language usage.

[![](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I'm%20fine%20with%20the%20Python%20library)](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I'm%20fine%20with%20the%20Python%20library/vote)
[![](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I'm%20fine%20with%20R%20wrapper)](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I'm%20fine%20with%20R%20wrapper/vote)
[![](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I%20would%20only%20use%20the%20methodology%20with%20a%20Stata%20wrapper)](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I%20would%20only%20use%20the%20methodology%20with%20a%20Stata%20wrapper/vote)
[![](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I%20would%20only%20use%20the%20methodology%20with%20a%20MATLAB%20wrapper)](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I%20would%20only%20use%20the%20methodology%20with%20a%20MATLAB%20wrapper/vote)
[![](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I%20would%20only%20use%20the%20methodology%20with%20a%20SAS%20wrapper)](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I%20would%20only%20use%20the%20methodology%20with%20a%20SAS%20wrapper/vote)
[![](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I%20would%20only%20use%20the%20methodology%20with%20an%20SPSS%20wrapper)](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I%20would%20only%20use%20the%20methodology%20with%20an%20SPSS%20wrapper/vote)


[LSTM neural networks](https://en.wikipedia.org/wiki/Long_short-term_memory) have been used for nowcasting [before](https://papers.nips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf), combining the strengths of artificial neural networks with a temporal aspect. However their use in nowcasting economic indicators remains limited, no doubt in part due to the difficulty of obtaining results in existing deep learning frameworks. This library seeks to streamline the process of obtaining results in the hopes of expanding the domains to which LSTM can be applied.

While neural networks are flexible and this framework may be able to get sensible results on levels, the model architecture was developed to nowcast growth rates of economic indicators. As such training inputs should ideally be stationary and seasonally adjusted.

Further explanation of the background problem can be found in [this UNCTAD research paper](https://unctad.org/system/files/official-document/ser-rp-2018d9_en.pdf). Further explanation and results can be found in [this](https://unctad.org/webflyer/economic-nowcasting-long-short-term-memory-artificial-neural-networks-lstm) UNCTAD research paper.

## R wrapper 
An R wrapper exists for this Python library: [https://github.com/dhopp1/nowcastLSTM](https://github.com/dhopp1/nowcastLSTM). Python and some Python libraries still need to be installed on your system, but full functionality from R can be obtained with the wrapper without any Python knowledge.