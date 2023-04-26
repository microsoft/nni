import * as React from 'react';
import { Outlet } from 'react-router-dom';
import { Stack, MessageBar, MessageBarType } from '@fluentui/react';
import { EXPERIMENT, TRIALS } from '@static/datamodel';
import NavCon from '@components/nav/Nav';
import { COLUMN } from '@static/const';
import { isManagerExperimentPage } from '@static/function';
import '@style/App.scss';
import '@style/common/common.scss';
import '@style/experiment/trialdetail/trialsDetail.scss';

const echarts = require('echarts/lib/echarts');
echarts.registerTheme('nni_theme', {
    color: '#3c8dbc'
});
export const NavContext = React.createContext({
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    changeInterval: (_val: number) => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    refreshPage: (): void => {}
});
export const AppContext = React.createContext({
    interval: 10, // sendons
    columnList: COLUMN,
    experimentUpdateBroadcast: 0,
    trialsUpdateBroadcast: 0,
    metricGraphMode: 'Maximize',
    bestTrialEntries: '10',
    maxDurationUnit: 'm',
    expandRowIDs: new Set(['']),
    expandRowIDsDetailTable: new Set(['']),
    selectedRowIds: [] as string[],
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    changeSelectedRowIds: (_val: string[]): void => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    changeColumn: (_val: string[]): void => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    changeMetricGraphMode: (_val: 'Maximize' | 'Minimize'): void => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    changeMaxDurationUnit: (_val: string): void => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    changeEntries: (_val: string): void => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    updateOverviewPage: () => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    updateDetailPage: () => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    changeExpandRowIDs: (_val: string, _type?: string): void => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    changeExpandRowIDsDetailTable: (_val: string, _type?: string): void => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    startTimer: () => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    closeTimer: (): void => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    refreshDetailTable: (): void => {}
});

interface AppState {
    interval: number;
    columnList: string[];
    experimentUpdateBroadcast: number;
    trialsUpdateBroadcast: number;
    maxDurationUnit: string;
    metricGraphMode: 'Maximize' | 'Minimize'; // tuner's optimize_mode filed
    isillegalFinal: boolean;
    expWarningMessage: string;
    bestTrialEntries: string; // for overview page: best trial entreis
    expandRowIDs: Set<string>; // for overview page: open row
    expandRowIDsDetailTable: Set<string>; // for overview page: open row
    selectedRowIds: string[]; // for detail page: selected trial - checkbox
    timerIdList: number[];
}

class App extends React.Component<{}, AppState> {
    private timerId = 0;

    constructor(props: {}) {
        super(props);
        this.state = {
            interval: 10, // sendons
            columnList: COLUMN,
            experimentUpdateBroadcast: 0,
            trialsUpdateBroadcast: 0,
            metricGraphMode: 'Maximize',
            maxDurationUnit: 'm',
            isillegalFinal: false,
            expWarningMessage: '',
            bestTrialEntries: '10',
            expandRowIDs: new Set(),
            expandRowIDsDetailTable: new Set(),
            selectedRowIds: [],
            timerIdList: []
        };
    }

    async componentDidMount(): Promise<void> {
        localStorage.removeItem('columns');
        localStorage.removeItem('paraColumns');
        await Promise.all([EXPERIMENT.init(), TRIALS.init()]);
        this.setState(state => ({
            experimentUpdateBroadcast: state.experimentUpdateBroadcast + 1,
            trialsUpdateBroadcast: state.trialsUpdateBroadcast + 1,
            metricGraphMode: EXPERIMENT.optimizeMode === 'minimize' ? 'Minimize' : 'Maximize'
        }));

        this.startTimer();
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
            bestTrialEntries,
            maxDurationUnit,
            expandRowIDs,
            expandRowIDsDetailTable,
            selectedRowIds
        } = this.state;
        if (experimentUpdateBroadcast === 0 || trialsUpdateBroadcast === 0) {
            return null;
        }
        const errorList = [
            { errorWhere: TRIALS.jobListError(), errorMessage: TRIALS.getJobErrorMessage() },
            { errorWhere: EXPERIMENT.experimentError(), errorMessage: EXPERIMENT.getExperimentMessage() },
            { errorWhere: EXPERIMENT.statusError(), errorMessage: EXPERIMENT.getStatusMessage() },
            { errorWhere: TRIALS.MetricDataError(), errorMessage: TRIALS.getMetricDataErrorMessage() },
            { errorWhere: TRIALS.latestMetricDataError(), errorMessage: TRIALS.getLatestMetricDataErrorMessage() },
            { errorWhere: TRIALS.metricDataRangeError(), errorMessage: TRIALS.metricDataRangeErrorMessage() }
        ];

        return (
            <React.Fragment>
                {isManagerExperimentPage() ? null : (
                    <Stack className='nni' style={{ minHeight: window.innerHeight }}>
                        <div className='header'>
                            <div className='headerCon'>
                                <NavContext.Provider
                                    value={{
                                        changeInterval: this.changeInterval,
                                        refreshPage: this.lastRefresh
                                    }}
                                >
                                    <NavCon />
                                </NavContext.Provider>
                            </div>
                        </div>
                        <Stack className='contentBox'>
                            <Stack className='content'>
                                {/* if api has error field, show error message */}
                                {errorList.map(
                                    (item, key) =>
                                        item.errorWhere && (
                                            <div key={key} className='warning'>
                                                <MessageBar messageBarType={MessageBarType.error}>
                                                    {item.errorMessage}
                                                </MessageBar>
                                            </div>
                                        )
                                )}
                                {isillegalFinal && (
                                    <div className='warning'>
                                        <MessageBar messageBarType={MessageBarType.warning}>
                                            {expWarningMessage}
                                        </MessageBar>
                                    </div>
                                )}
                                {/* <AppContext.Provider */}
                                <AppContext.Provider
                                    value={{
                                        interval,
                                        columnList,
                                        changeColumn: this.changeColumn,
                                        experimentUpdateBroadcast,
                                        trialsUpdateBroadcast,
                                        metricGraphMode,
                                        maxDurationUnit,
                                        bestTrialEntries,
                                        changeMaxDurationUnit: this.changeMaxDurationUnit,
                                        changeMetricGraphMode: this.changeMetricGraphMode,
                                        changeEntries: this.changeEntries,
                                        expandRowIDs,
                                        expandRowIDsDetailTable,
                                        selectedRowIds,
                                        changeSelectedRowIds: this.changeSelectedRowIds,
                                        changeExpandRowIDs: this.changeExpandRowIDs,
                                        changeExpandRowIDsDetailTable: this.changeExpandRowIDsDetailTable,
                                        updateOverviewPage: this.updateOverviewPage,
                                        updateDetailPage: this.updateDetailPage, // update current record without fetch api
                                        refreshDetailTable: this.refreshDetailTable, // update record with fetch api
                                        startTimer: this.startTimer,
                                        closeTimer: this.closeTimer
                                    }}
                                >
                                    <Outlet />
                                    {this.props.children}
                                </AppContext.Provider>
                            </Stack>
                        </Stack>
                    </Stack>
                )}
            </React.Fragment>
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

        // experiment status and /trial-jobs api's status could decide website update
        if (['DONE', 'ERROR', 'STOPPED', 'VIEWED'].includes(EXPERIMENT.status) || TRIALS.jobListError()) {
            // experiment finished, refresh once more to ensure consistency
            this.setState(() => ({ interval: 0 }));
            this.closeTimer();
            return;
        }

        this.startTimer();
    };

    public lastRefresh = async (): Promise<void> => {
        await EXPERIMENT.update();
        await TRIALS.update(true);
        this.setState(state => ({
            experimentUpdateBroadcast: state.experimentUpdateBroadcast + 1,
            trialsUpdateBroadcast: state.trialsUpdateBroadcast + 1
        }));
    };

    public changeInterval = (interval: number): void => {
        this.setState(() => ({ interval: interval })); // reset interval val
        this.closeTimer(); // close page auto refresh
        if (interval !== 0) {
            this.refresh();
        }
    };

    public changeColumn = (columnList: string[]): void => {
        this.setState({ columnList: columnList });
    };

    // for succeed table in the overview page
    public changeExpandRowIDs = (id: string, type?: string): void => {
        const currentExpandRowIDs = this.state.expandRowIDs;

        if (!currentExpandRowIDs.has(id)) {
            currentExpandRowIDs.add(id);
        } else {
            if (!(type !== undefined && type === 'chart')) {
                currentExpandRowIDs.delete(id);
            }
        }

        this.setState({ expandRowIDs: currentExpandRowIDs });
    };

    // for details table in the detail page
    public changeExpandRowIDsDetailTable = (id: string): void => {
        const currentExpandRowIDs = this.state.expandRowIDsDetailTable;

        if (!currentExpandRowIDs.has(id)) {
            currentExpandRowIDs.add(id);
        } else {
            currentExpandRowIDs.delete(id);
        }

        this.setState({ expandRowIDsDetailTable: currentExpandRowIDs });
    };

    public changeSelectedRowIds = (val: string[]): void => {
        this.setState({ selectedRowIds: val });
    };
    public changeMetricGraphMode = (val: 'Maximize' | 'Minimize'): void => {
        this.setState({ metricGraphMode: val });
    };

    // overview best trial module
    public changeEntries = (entries: string): void => {
        this.setState({ bestTrialEntries: entries });
    };

    // overview max duration unit
    public changeMaxDurationUnit = (unit: string): void => {
        this.setState({ maxDurationUnit: unit });
    };

    public updateOverviewPage = (): void => {
        this.setState(state => ({
            experimentUpdateBroadcast: state.experimentUpdateBroadcast + 1
        }));
    };

    public updateDetailPage = async (): Promise<void> => {
        this.setState(state => ({
            trialsUpdateBroadcast: state.trialsUpdateBroadcast + 1
        }));
    };

    // fetch api to update table record data
    public refreshDetailTable = async (): Promise<void> => {
        await TRIALS.update(true);
        this.setState(state => ({
            trialsUpdateBroadcast: state.trialsUpdateBroadcast + 1
        }));
    };

    // start to refresh page automatically
    public startTimer = (): void => {
        this.timerId = window.setTimeout(this.refresh, this.state.interval * 1000);
        const storeTimerList = this.state.timerIdList;
        storeTimerList.push(this.timerId);
        this.setState(() => ({ timerIdList: storeTimerList }));
    };

    public closeTimer = (): void => {
        const { timerIdList } = this.state;
        timerIdList.forEach(item => {
            window.clearTimeout(item);
        });
    };
}

export default App;
