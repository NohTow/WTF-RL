# WTF-RL
This is the repository for the code of the "Distinctive Image Captioning: Leveraging Ground Truth Captions in CLIP Guided Reinforcement Learning" paper.

It explores how ground truth captions can be leveraged to train image captioning models using cross-modal rewards in a reinforcement learning training scheme, where they are not needed.

We show that ground truth captions can be leveraged to ground the training to the original distribution. First, they can be used as samples for the reinforcement learning objective, resulting in a teacher forcing objective weighted by the reward. This objective train the model to reproduce human samples while focusing on the most distinctive ones, matching the traditionnal RL objective.\
Second, they can be used to train a discriminator that serve as a regularization term to the generator to further ground the generator to the human distribution. This grounding is a first step towards training both models jointly by limiting the inherent drifting from the human language due to the cooperative of the two models.\
Finally, we introduce a contrastive rewards, that consider every element in the batch as baselines for the reward, letting the generator learn from the best sequences only. This contrastive reward, in addition to be very cheap to compute, natively consider both cross-modal retrieval directions, enabling to produce captions that are very descriptive of the input image and this image only.

## Setup :wrench:
On a Python 3 installation (tested in 3.8), install the dependencies defined in the requirement.txt file using

    pip install -r requirements.txt

The results of the paper are based on the [OFA model](https://github.com/OFA-Sys/OFA), so please follow the [installation guide](https://github.com/OFA-Sys/OFA/blob/main/transformers.md) to install the transformers library version that implements OFA to reproduce results.

If you just want to experiment with the approach, we also give a version working with the BLIP model, which is natively in the transformers library and that can be installed using

    pip install transformers


## License
```
BSD 3-Clause-Attribution License

Copyright (c) 2022, IMATAG and CNRS

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

4. Redistributions of any form whatsoever must retain the following acknowledgment: 
   'This product includes software developed by IMATAG and CNRS'

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
``` 
