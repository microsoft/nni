import * as React from 'react';
import { Stack } from 'office-ui-fabric-react';
import { COLUMN } from './static/const';
import { EXPERIMENT, TRIALS } from './static/datamodel';
import NavCon from './components/NavCon';
import MessageInfo from './components/Modals/MessageInfo';
import './App.scss';

interface AppState {
    interval: number;
    columnList: string[];
    experimentUpdateBroadcast: number;
    trialsUpdateBroadcast: number;
    metricGraphMode: 'max' | 'min'; // tuner's optimize_mode filed
    isillegalFinal: boolean;
    expWarningMessage: string;
    bestTrialEntries: string; // for overview page: best trial entreis
    isUpdate: boolean;
}

class App extends React.Component<{}, AppState> {
    private timerId!: number | undefined;
    private dataFormatimer!: number;
    private firstLoad: boolean = false; // when click refresh selector options
    
    constructor(props: {}) {
        super(props);
        this.state = {
            interval: 10, // sendons
            columnList: COLUMN,
            experimentUpdateBroadcast: 0,
            trialsUpdateBroadcast: 0,
            metricGraphMode: 'max',
            isillegalFinal: false,
            expWarningMessage: '',
            bestTrialEntries: '10',
            isUpdate: true
        };
    }

    async componentDidMount(): Promise<void> {
        await Promise.all([EXPERIMENT.init(), TRIALS.init()]);
        this.setState(state => ({ 
            experimentUpdateBroadcast: state.experimentUpdateBroadcast + 1, 
            trialsUpdateBroadcast: state.trialsUpdateBroadcast + 1,
            metricGraphMode: (EXPERIMENT.optimizeMode === 'minimize' ? 'min' : 'max')
        }));
        this.timerId = window.setTimeout(this.refresh, this.state.interval * 100);
        // final result is legal
        // get a succeed trial，see final result data's format
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        this.dataFormatimer = window.setInterval(this.getFinalDataFormat, this.state.interval * 1000);
    }

    getFinalDataFormat = (): void => {
        for (let i = 0; this.state.isillegalFinal === false; i++) {
            if (TRIALS.succeededTrials()[0] !== undefined && TRIALS.succeededTrials()[0].final !== undefined) {
                // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
                const oneSucceedTrial = JSON.parse(JSON.parse(TRIALS.succeededTrials()[0].final!.data));
                if (typeof oneSucceedTrial === 'number' || oneSucceedTrial.hasOwnProperty('default')) {
                    window.clearInterval(this.dataFormatimer);
                    break;
                } else {
                    // illegal final data
                    this.setState(() => ({
                        isillegalFinal: true,
                        expWarningMessage:
                            'WebUI support final result as number and dictornary includes default keys, your experiment final result is illegal, please check your data.'
                    }));
                    window.clearInterval(this.dataFormatimer);
                }
            } else {
                break;
            }
        }
    };

    changeInterval = (interval: number): void => {
        window.clearTimeout(this.timerId);
        if (interval === 0) {
            return;
        }
        // setState will trigger page refresh at once.
        // setState is asyc, interval not update to (this.state.interval) at once.
        this.setState({ interval }, () => {
            this.firstLoad = true;
            this.refresh();
        });
    };

    // TODO: use local storage
    changeColumn = (columnList: string[]): void => {
        this.setState({ columnList: columnList });
    };

    changeMetricGraphMode = (val: 'max' | 'min'): void => {
        this.setState({ metricGraphMode: val });
    };

    // overview best trial module
    changeEntries = (entries: string): void => {
        this.setState({ bestTrialEntries: entries });
    };

    shouldComponentUpdate(nextProps: any, nextState: AppState): boolean {
        
        if(!(nextState.isUpdate || nextState.isUpdate === undefined)){
            nextState.isUpdate = true;
            return false;
        }
        return true;
    }

    render(): React.ReactNode {
        const {
            interval,
            columnList,
            experimentUpdateBroadcast,
            trialsUpdateBroadcast,
            metricGraphMode,
            isillegalFinal,
            expWarningMessage,
            bestTrialEntries
        } = this.state;
        if (experimentUpdateBroadcast === 0 || trialsUpdateBroadcast === 0) {
            return null; // TODO: render a loading page
        }
        const errorList = [
            { errorWhere: TRIALS.jobListError(), errorMessage: TRIALS.getJobErrorMessage() },
            { errorWhere: EXPERIMENT.experimentError(), errorMessage: EXPERIMENT.getExperimentMessage() },
            { errorWhere: EXPERIMENT.statusError(), errorMessage: EXPERIMENT.getStatusMessage() },
            { errorWhere: TRIALS.MetricDataError(), errorMessage: TRIALS.getMetricDataErrorMessage() },
            { errorWhere: TRIALS.latestMetricDataError(), errorMessage: TRIALS.getLatestMetricDataErrorMessage() },
            { errorWhere: TRIALS.metricDataRangeError(), errorMessage: TRIALS.metricDataRangeErrorMessage() }
        ];

        const reactPropsChildren = React.Children.map(this.props.children, child =>
            React.cloneElement(child as React.ReactElement<any>, {
                interval,
                columnList,
                changeColumn: this.changeColumn,
                experimentUpdateBroadcast,
                trialsUpdateBroadcast,
                metricGraphMode,
                changeMetricGraphMode: this.changeMetricGraphMode,
                bestTrialEntries,
                changeEntries: this.changeEntries
            })
        );

        return (
            <Stack className='nni' style={{ minHeight: window.innerHeight }}>
                <div className='header'>
                    <div className='headerCon'>
                        <NavCon changeInterval={this.changeInterval} refreshFunction={this.lastRefresh} />
                    </div>
                </div>
                <Stack className='contentBox'>
                    <Stack className='content'>
                        {/* if api has error field, show error message */}
                        {errorList.map(
                            (item, key) =>
                                item.errorWhere && (
                                    <div key={key} className='warning'>
                                        <MessageInfo info={item.errorMessage} typeInfo='error' />
                                    </div>
                                )
                        )}
                        {isillegalFinal && (
                            <div className='warning'>
                                <MessageInfo info={expWarningMessage} typeInfo='warning' />
                            </div>
                        )}
                        {reactPropsChildren}
                    </Stack>
                </Stack>
            </Stack>
        );
    }

    private refresh = async (): Promise<void> => {
        // resolve this question: 10s -> 20s, page refresh twice.
        // only refresh this page after clicking the refresh options
        if (this.firstLoad !== true) {
            const [experimentUpdated, trialsUpdated] = await Promise.all([EXPERIMENT.update(), TRIALS.update()]);
            if (experimentUpdated) {
                this.setState(state => ({ experimentUpdateBroadcast: state.experimentUpdateBroadcast + 1 }));
            }
            if (trialsUpdated) {
                this.setState(state => ({ trialsUpdateBroadcast: state.trialsUpdateBroadcast + 1 }));
            }
        } else {
            this.firstLoad = false;
        }

        // experiment status and /trial-jobs api's status could decide website update
        if (['DONE', 'ERROR', 'STOPPED'].includes(EXPERIMENT.status) || TRIALS.jobListError()) {
            // experiment finished, refresh once more to ensure consistency
            this.setState(() => ({ interval: 0, isUpdate: false }));
            return;
        }

        this.timerId = window.setTimeout(this.refresh, this.state.interval * 1000);
    };

    public async lastRefresh(): Promise<void> {
        await EXPERIMENT.update();
        await TRIALS.update(true);
        this.setState(state => ({ experimentUpdateBroadcast: state.experimentUpdateBroadcast + 1, trialsUpdateBroadcast: state.trialsUpdateBroadcast + 1 }));
    }
}

export default App;
