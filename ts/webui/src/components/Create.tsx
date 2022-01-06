import * as React from 'react';
// import { Stack, Icon, Dropdown, DefaultButton } from '@fluentui/react';
import { Stack } from '@fluentui/react';
import { EXPERIMENT } from '../static/datamodel';
import { AppContext } from '../App';
import { Title } from './create/Title';
// import { ExpDuration } from './create/count/ExpDuration';
// import { ExpDurationContext } from './create/count/ExpDurationContext';
import { TitleContext } from './create/TitleContext';
// import { itemStyleSucceed, entriesOption } from './create/overviewConst';
import '../static/style/create/overview.scss';
import '../static/style/create/topTrial.scss';
import '../static/style/logPath.scss';
import { TextField } from '@fluentui/react/lib/TextField';
import { IStackProps, IStackStyles } from '@fluentui/react/lib/Stack';
import { IStackTokens } from '@fluentui/react';
import { DefaultButton } from '@fluentui/react/lib/Button';
// import { IStackStyles } from '@fluentui/react/lib/Stack';
// import { lorem } from '@fluentui/example-data';

const stackStyles: Partial<IStackStyles> = { root: { width: 650 } };
const stackTokens = { childrenGap: 50 };
const buttonTokens: IStackTokens = { childrenGap: 40 };
const local_mode_txt: string =
    'experimentName: MNIST \n\
    trialCommand: python mnist.py \n\
    trialCodeDirectory: . \n\
    trialGpuNumber: 1 \n\
    trialConcurrency: 2 \n\
    maxExperimentDuration: 24h \n\
    maxTrialNumber: 100 \n\
    tuner: \n\
        \tname: TPE \n\
        \tclassArgs: \n\
            \t\toptimize_mode: maximize \n\
    trainingService: \n\
        \tplatform: local \n\
        \tuseActiveGpu: True';
const search_space_txt: string =
    '{ \n\
    "batch_size": { \n\
      \t"_type": "choice", \n\
      \t"_value": [16, 32, 64, 128] \n\
    }, \n\
    "hidden_size": { \n\
      \t"_type": "choice", \n\
      \t"_value": [128, 256, 512, 1024] \n\
    }, \n\
    "lr": { \n\
      \t"_type": "choice", \n\
      \t"_value": [0.0001, 0.001, 0.01, 0.1] \n\
    }, \n\
    "momentum": { \n\
      \t"_type": "uniform", \n\
      \t"_value": [0, 1] \n\
    } \n\
  }';
const experiment_txt: string =
    'experimentName: Optional[str] \n\
    trialCommand: str \n\
    trialCodeDirectory: str, default . \n\
    trialConcurrency: int \n\
    trialGpuNumber: Optional[int] \n\
    maxExperimentDuration: Optional[str], 10m, 0.5h \n\
    maxTrialNumber: Optional[int] \n\
    nniManagerIp: Optional[str] \n\
    useAnnottaion: Optional[bool] \n\
    debug: str \n\
    logLevel: Optional[str]\n\
    experimentWorkingDirectory: Optional[str]\n\
    tunerGpuIndices: Optional[list[int] | str | int] \n\
    tuner: Optional AlgorithmConfig \n\
    trainingService: TrainingServiceConfig \n\
    sharedStorage: Optional SharedStorageConfig \n\
    AlgorithmConfig: \n\
    name:  \n\
    classArgs: \n\
    TrainingServiceConfig: \n\
    LocalConfig: \n\
    platform: const str \n\
    useActiveGpu: \n\
    maxTrialNumberPerGpu: int, default 1 \n\
    gpuIndices: \n\
    RemoteConfig: \n\
    platform: \n\
    machineList: \n\
        \tRemoteMachineConfig: \n\
        \thost: \n\
        \tport: \n\
        \tuser: \n\
        \tpassword: Optional[str] \n\
        \tsshKeyFile: \n\
        \tsshPassphrase: \n\
        \tuseActiveGpu: \n\
        \tmaxTrialNumberPerGpu: \n\
        \tgpuIndices: \n\
        \tpythonPath: \n\
    reuseMode: ';
const columnProps: Partial<IStackProps> = {
    tokens: { childrenGap: 15 },
    styles: { root: { width: 350 } }
};

function _alertClicked(): void {
    alert('Submit successfully!');
}

interface CreateState {
    trialConcurrency: number;
}

export const BestMetricContext = React.createContext({
    bestAccuracy: 0
});

class Create extends React.Component<{}, CreateState> {
    static contextType = AppContext;
    context!: React.ContextType<typeof AppContext>;

    constructor(props) {
        super(props);
        this.state = {
            trialConcurrency: EXPERIMENT.trialConcurrency
        };
    }

    // clickMaxTop = (event: React.SyntheticEvent<EventTarget>): void => {
    //     event.stopPropagation();
    //     // #999 panel active bgcolor; #b3b3b3 as usual
    //     const { changeMetricGraphMode } = this.context;
    //     changeMetricGraphMode('max');
    // };

    // clickMinTop = (event: React.SyntheticEvent<EventTarget>): void => {
    //     event.stopPropagation();
    //     const { changeMetricGraphMode } = this.context;
    //     changeMetricGraphMode('min');
    // };

    // updateEntries = (event: React.FormEvent<HTMLDivElement>, item: any): void => {
    //     if (item !== undefined) {
    //         this.context.changeEntries(item.key);
    //     }
    // };

    render(): React.ReactNode {
        // const bestTrials = this.findBestTrials();
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        // const bestAccuracy = bestTrials.length > 0 ? bestTrials[0].accuracy! : NaN;
        // const execDuration = EXPERIMENT.profile.execDuration;

        return (
            <AppContext.Consumer>
                {(): React.ReactNode => {
                    // const { metricGraphMode, bestTrialEntries } = value;
                    // const maxActive = metricGraphMode === 'max' ? 'active' : '';
                    // const minActive = metricGraphMode === 'min' ? 'active' : '';
                    return (
                        <div className='overview'>
                            <div className='overviewWrapper'>
                                {/* exp params */}
                                <div className='overviewBasicInfo'>
                                    <Stack horizontal tokens={stackTokens} styles={stackStyles}>
                                        <Stack>
                                            <TitleContext.Provider value={{ text: 'Local Mode', icon: 'AutoRacing' }}>
                                                <Title />
                                            </TitleContext.Provider>
                                        </Stack>

                                        <Stack {...columnProps}>
                                            <TextField label='' multiline rows={16} defaultValue={local_mode_txt} />
                                        </Stack>
                                    </Stack>
                                </div>
                                {/* duration & trial numbers */}
                                <div className='duration'>
                                    <Stack horizontal tokens={stackTokens} styles={stackStyles}>
                                        <Stack>
                                            <TitleContext.Provider value={{ text: 'Search Space', icon: 'AutoRacing' }}>
                                                <Title />
                                            </TitleContext.Provider>
                                        </Stack>

                                        <Stack {...columnProps}>
                                            <TextField label='' multiline rows={16} defaultValue={search_space_txt} />
                                        </Stack>
                                    </Stack>
                                </div>
                                {/* table */}
                                <div className='overviewBestMetric'>
                                    <Stack horizontal tokens={stackTokens} styles={stackStyles}>
                                        <Stack>
                                            <TitleContext.Provider value={{ text: 'Experiment', icon: 'AutoRacing' }}>
                                                <Title />
                                            </TitleContext.Provider>
                                        </Stack>

                                        <Stack {...columnProps}>
                                            <TextField label='' multiline rows={32} defaultValue={experiment_txt} />
                                        </Stack>
                                    </Stack>
                                </div>
                                {/* submit button */}
                                <div className='submit_button'>
                                    <Stack horizontal tokens={buttonTokens}>
                                        <DefaultButton text='Submit' onClick={_alertClicked} allowDisabledFocus />
                                    </Stack>
                                </div>
                            </div>
                        </div>
                    );
                }}
            </AppContext.Consumer>
        );
    }
}

export default Create;
