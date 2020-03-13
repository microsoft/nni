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
    isilLegalFinal: boolean;
    expWarningMessage: string;
}

class App extends React.Component<{}, AppState> {
    private timerId!: number | null;

    constructor(props: {}) {
        super(props);
        this.state = {
            interval: 10, // sendons
            columnList: COLUMN,
            experimentUpdateBroadcast: 0,
            trialsUpdateBroadcast: 0,
            metricGraphMode: 'max',
            isilLegalFinal: false,
            expWarningMessage: ''
        };
    }

    async componentDidMount(): Promise<void> {
        await Promise.all([EXPERIMENT.init(), TRIALS.init()]);
        this.setState(state => ({ experimentUpdateBroadcast: state.experimentUpdateBroadcast + 1 }));
        this.setState(state => ({ trialsUpdateBroadcast: state.trialsUpdateBroadcast + 1 }));
        this.timerId = window.setTimeout(this.refresh, this.state.interval * 1000);
        this.setState({ metricGraphMode: (EXPERIMENT.optimizeMode === 'minimize' ? 'min' : 'max') });
        // final result is legal
        // 选一条succeed trial，查看final result格式是否支持
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        window.setInterval(this.test, this.state.interval * 1000);
        
    }

    test = () => {
        console.info('例行检查'); // eslint-disable-line
        for(let i = 0; this.state.isilLegalFinal === false; i++){
            if(TRIALS.succeededTrials()[0] !== undefined && TRIALS.succeededTrials()[0].final !== undefined){
                const oneSucceedTrial = JSON.parse(TRIALS.succeededTrials()[0].final!.data);
                if (typeof oneSucceedTrial === 'number' || oneSucceedTrial.hasOwnProperty('default')) {
                    return;
                } else {
                    console.info('数据不合常理'); // eslint-disable-line
                    // 非法
                    this.setState(() => ({
                        isilLegalFinal: true,
                        expWarningMessage: 'WebUI support final result as number and dictornary includes default keys, your experiment final result is illegal, please check your data.'
                    }));
                }
            }
        }
    }

    changeInterval = (interval: number): void => {
        this.setState({ interval });
        if (this.timerId === null && interval !== 0) {
            window.setTimeout(this.refresh);
        } else if (this.timerId !== null && interval === 0) {
            window.clearTimeout(this.timerId);
        }
    }

    // TODO: use local storage
    changeColumn = (columnList: string[]): void => {
        this.setState({ columnList: columnList });
    }

    changeMetricGraphMode = (val: 'max' | 'min'): void => {
        this.setState({ metricGraphMode: val });
    }

    render(): React.ReactNode {
        const { interval, columnList, experimentUpdateBroadcast, trialsUpdateBroadcast,
        metricGraphMode, isilLegalFinal, expWarningMessage } = this.state;
        if (experimentUpdateBroadcast === 0 || trialsUpdateBroadcast === 0) {
            return null;  // TODO: render a loading page
        }
        const reactPropsChildren = React.Children.map(this.props.children, child =>
            React.cloneElement(
                child as React.ReactElement<any>, {
                interval,
                columnList, changeColumn: this.changeColumn,
                experimentUpdateBroadcast,
                trialsUpdateBroadcast,
                metricGraphMode, changeMetricGraphMode: this.changeMetricGraphMode
            })
        );

        return (
            <Stack className="nni" style={{ minHeight: window.innerHeight }}>
                <div className="header">
                    <div className="headerCon">
                        <NavCon changeInterval={this.changeInterval} refreshFunction={this.lastRefresh} />
                    </div>
                </div>
                <Stack className="contentBox">
                    <Stack className="content">
                        {isilLegalFinal && <div className="warning">
                            <MessageInfo info={expWarningMessage} typeInfo="warning" />
                        </div>}
                        {reactPropsChildren}
                    </Stack>
                </Stack>
            </Stack>
        );
    }

    private refresh = async (): Promise<void> => {
        const [experimentUpdated, trialsUpdated] = await Promise.all([EXPERIMENT.update(), TRIALS.update()]);
        if (experimentUpdated) {
            this.setState(state => ({ experimentUpdateBroadcast: state.experimentUpdateBroadcast + 1 }));
        }
        if (trialsUpdated) {
            this.setState(state => ({ trialsUpdateBroadcast: state.trialsUpdateBroadcast + 1 }));
        }

        if (['DONE', 'ERROR', 'STOPPED'].includes(EXPERIMENT.status)) {
            // experiment finished, refresh once more to ensure consistency
            if (this.state.interval > 0) {
                this.setState({ interval: 0 });
                this.lastRefresh();
            }

        } else if (this.state.interval !== 0) {
            this.timerId = window.setTimeout(this.refresh, this.state.interval * 1000);
        }
    }

    public async lastRefresh(): Promise<void> {
        await EXPERIMENT.update();
        await TRIALS.update(true);
        this.setState(state => ({ experimentUpdateBroadcast: state.experimentUpdateBroadcast + 1 }));
        this.setState(state => ({ trialsUpdateBroadcast: state.trialsUpdateBroadcast + 1 }));
    }
}

export default App;


