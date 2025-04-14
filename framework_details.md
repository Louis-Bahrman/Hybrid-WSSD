# Framework

This document details the training framework and explains how to add your own models and datasets.

## Training

This framework uses [Pytorch Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html). 

Training scripts for each supervision variant are provided in COMPLETE ME.

### Overview.

The whole training procedure is defined in `model.joint_model.JointModel`.
This LightningModule is composed of 3 main components:

- Speech model (Dereverberation)
- Reverb model (reverberation parameters analysis and RIR synthesis)
- Joint Loss (convolutive model and reverberation matching loss)

Config files are located in `config/speech_models/` and `config/rir_models/` for speech model and reverb model respectively.
See below how to add new components.

### Configuration of `JointModel` for strong, weak, or un-supervision

The loss on which the model is trained (and hence the supervision variant) depends on the arguments to initialize `JointModel`, especially `joint_loss_module`.
This behaviour corresponds to `JointModel.training_step` and should not be modified when adding a new speech or reverb model (to maintain backward compatibility).

- If `joint_loss_module` is defined, and ŝ and ĥ returned by the `speech_model` and `reverb_model` respectively are not None, 
the training loss is the output of `joint_loss_module.forward`.
In that case, all other losses (computed by `internal_loss` in both `speech_model` and `reverb_model`) are logged but not backpropagated.

- If `joint_loss_module` is not defined, then, one and only one of (`speech_model`, `reverb_model`) should be defined. 
If both `speech_model` and `reverb_model` are defined, there is an ambiguity on which loss to backpropagate and an error is raised.
In case `speech_model` is defined, the backpropagated loss is the output of `speech_model.internal_loss`.
In case `reverb_model` is defined, the backpropagated loss is the output of `reverb_model.internal_loss`.

## Adding a new dataset

1. Define a new subclass of `datasets.AudioDatasetConvolvedWithRirDatasetDatamodule`
2. Define a new `split_audio_dataset` method to perform train-val-test-splitting of audio (independently from train-val-test-splitting of RIRs). An example is provided in `WSJSimulatedRirDataModule`.
3. By default, the behaviour of `AudioDatasetConvolvedWithRirDatasetDatamodule` is to use convolve a dry signal with an RIR at runtime. If you don't want this behaviour, you have to redefine `setup` and `__init__`. An example is provided in `EarsReverbDataModule`.


## Adding a new model

Some abstract classes are provided in `model.utils.abs_models.py`. 
Please inherit from them when adding a new model. Detailed explanations are given below. 

Each component (speech model, reverb model, or joint loss) is inherited from a `FirstLevelModule`, which handles metrics computation, logging, and STFT and ISTFT transforms (defined in `model.utils.default_stft_istft`).
Then, for both speech and reverb models the training and validation procedure is always the same, and defined in `AbsSpeechOrReverbModel`.
When inheriting from this class, (and not using an oracle model) you shouldn't modify `training_step` nor `validation_step` but  define the following methods:

- `internal_loss`: Internal loss of the model, when it is used in standalone, without being trained in a reverberation-aware setting. For instance, `FullSubNet.internal_loss` computes the optimal complex mask and compares it with the predicted complex mask.
- `forward`: takes wet signal as input and returns the prediction, and all temporary tensors which are used to compute the internal loss.
For instance, `FullSubNet.forward` returns both the predicted mask and elements to compute the optimal mask.
- `get_time`: From the output of forward, returns the time-domain prediction (either speech or reverb signal). Defaults to returning the output of forward itself
- `get_stft`: From the output of forward, returns the STFT-domain prediction. Defaults to returning the STFT of the output of forward. Please use `default_stft_module` for the STFT parameters to be the same as the input of the joint-loss.
- `crop input to target` argument in `__init__`: Whether the wet input should be cropped or  to the target length. Should be `True` for speech models and 
`False` for reverb models.

You should not modify `JointModel` when adding a new speech model, reverb model, or joint loss, but instead create a subclass of the corresponding model. The use-cases are detailed below:

### Speech model

- Oracle: Oracle model defined in `OracleSpeechModel`. You should normally not inherit from this class. Training step just returns the target dry signal, and no Loss.
- Learnt: Inherit from `AbsSpeechModel` and define all methods mentionned above.

### Reverb model

- Oracle: Oracle model defined in `OracleReverbModel`. You should normally not inherit from this class. Training step just returns the target RIR, and no Loss.
- Oracle parameters: Inherit from this class when the RIR model does an analysis-synthesis of the RIR (example: Polack synthesis with oracle parameters). The only method you should implement when inheriting from this class is `convert_rir`, that takes as input a time-domain RIR and returns a time-domain synthesized RIR
- Learnt: Inherit from `AbsReverbModel` and define all methods mentionned above.

### Joint loss

To define a joint loss (Convolutive model and reverberation-matching loss) follow the following steps:

1. Inherit from `AbsJointLossModule`
2. Set `SPEECH_INPUT_DOMAIN` and `REVERB_INPUT_DOMAIN`, which are attributes stating whether the input of ŝ and ĥ should be in time or time-frequency domain.
Those attributes should be constant and are only read when instantiating `speech_model` and `reverb_model` in `JointModel.__init__`.
4. Define the `forward` method, that takes as input ŝ, ĥ, s, h and y and outputs a loss. 
5. Define the `validation_step`, to include logging of intermediate tensors if necessary. Otherwise just return the output of `training_step`.

See `model.joint_losses.rereverberation_loss.TimeFrequencyRereverberationLoss` for an example.
