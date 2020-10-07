import * as React from 'react';
import { Stack, Icon, Dropdown, DefaultButton } from '@fluentui/react';
import { EXPERIMENT, TRIALS } from '../static/datamodel';
import { Trial } from '../static/model/trial';
import { AppContext } from '../App';
import { Title } from './overview/Title';
import SuccessTable from './overview/table/SuccessTable';
import Accuracy from './overview/Accuracy';
import { TrialConfigButton } from './overview/config/TrialConfigButton';
import { ReBasicInfo } from './overview/experiment/BasicInfo';
import { ExpDuration } from './overview/count/ExpDuration';
import { ExpDurationContext } from './overview/count/ExpDurationContext';
import { TrialCount } from './overview/count/TrialCount';
import { Command } from './overview/experiment/Command';
import { TitleContext } from './overview/TitleContext';
import { itemStyle1, itemStyleSucceed, itemStyle2, entriesOption } from './overview/overviewConst';
import '../static/style/overview/overview.scss';
import '../static/style/logPath.scss';

interface OverviewState {
    trialConcurrency: number;
}

export const BestMetricContext = React.createContext({
    bestAccuracy: 0
});

class Overview extends React.Component<{}, OverviewState> {
    static contextType = AppContext;
    context!: React.ContextType<typeof AppContext>;

    constructor(props) {
        super(props);
        this.state = {
            trialConcurrency: EXPERIMENT.trialConcurrency
        };
    }

    clickMaxTop = (event: React.SyntheticEvent<EventTarget>): void => {
        event.stopPropagation();
        // #999 panel active bgcolor; #b3b3b3 as usual
        const { changeMetricGraphMode } = this.context;
        changeMetricGraphMode('max');
    };

    clickMinTop = (event: React.SyntheticEvent<EventTarget>): void => {
        event.stopPropagation();
        const { changeMetricGraphMode } = this.context;
        changeMetricGraphMode('min');
    };

    updateEntries = (event: React.FormEvent<HTMLDivElement>, item: any): void => {
        if (item !== undefined) {
            this.context.changeEntries(item.key);
        }
    };

    render(): React.ReactNode {
        const bestTrials = this.findBestTrials();
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        const bestAccuracy = bestTrials.length > 0 ? bestTrials[0].accuracy! : NaN;
        const accuracyGraphData = this.generateAccuracyGraph(bestTrials);
        const noDataMessage = bestTrials.length > 0 ? '' : 'No data';

        const maxExecDuration = EXPERIMENT.profile.params.maxExecDuration;
        const execDuration = EXPERIMENT.profile.execDuration;
        return (
            <AppContext.Consumer>
                {(value): React.ReactNode => {
                    const { metricGraphMode, bestTrialEntries, updateOverviewPage } = value;
                    const maxActive = metricGraphMode === 'max' ? 'active' : '';
                    const minActive = metricGraphMode === 'min' ? 'active' : '';
                    return (
                        <div className='overview'>
                            {/* search space & config */}
                            <TrialConfigButton />
                            <div className='wrapper'>
                                {/* exp params */}
                                <div className='box1'>
                                    <TitleContext.Provider value={{ text: 'Experiment', icon: 'AutoRacing' }}>
                                        <Title />
                                    </TitleContext.Provider>
                                    <BestMetricContext.Provider value={{ bestAccuracy: bestAccuracy }}>
                                        <ReBasicInfo />
                                    </BestMetricContext.Provider>
                                </div>
                                {/* duration & trial numbers */}
                                <div className='box2'>
                                    <div className='box4'>
                                        <TitleContext.Provider value={{ text: 'Duration', icon: 'Timer' }}>
                                            <Title />
                                        </TitleContext.Provider>
                                        <ExpDurationContext.Provider
                                            value={{ maxExecDuration, execDuration, updateOverviewPage }}
                                        >
                                            <ExpDuration />
                                        </ExpDurationContext.Provider>
                                    </div>
                                    <div className='empty' />
                                    <div className='box7'>
                                        <TitleContext.Provider value={{ text: 'Trial numbers', icon: 'NumberSymbol' }}>
                                            <Title />
                                        </TitleContext.Provider>
                                        <ExpDurationContext.Provider
                                            value={{ maxExecDuration, execDuration, updateOverviewPage }}
                                        >
                                            <TrialCount />
                                        </ExpDurationContext.Provider>
                                    </div>
                                </div>
                                {/* table */}
                                <div className='box3'>
                                    <Stack horizontal>
                                        <div style={itemStyleSucceed}>
                                            <TitleContext.Provider value={{ text: 'Top trials', icon: 'BulletedList' }}>
                                                <Title />
                                            </TitleContext.Provider>
                                        </div>
                                        <div className='topTrialTitle'>
                                            {/* <Stack horizontal horizontalAlign='space-between'> */}
                                            <Stack horizontal horizontalAlign='end'>
                                                <DefaultButton
                                                    onClick={this.clickMaxTop}
                                                    className={maxActive}
                                                    styles={{ root: { minWidth: 70, padding: 0 } }}
                                                >
                                                    <Icon iconName='Market' />
                                                    <span className='max'>Max</span>
                                                </DefaultButton>
                                                <div className='mincenter'>
                                                    <DefaultButton
                                                        onClick={this.clickMinTop}
                                                        className={minActive}
                                                        styles={{ root: { minWidth: 70, padding: 0 } }}
                                                    >
                                                        <Icon iconName='MarketDown' />
                                                        <span className='max'>Min</span>
                                                    </DefaultButton>
                                                </div>
                                                <div>
                                                    <Stack horizontal>
                                                        <div className='chooseEntry'>Display top</div>
                                                        <div>
                                                            <Dropdown
                                                                selectedKey={bestTrialEntries}
                                                                options={entriesOption}
                                                                onChange={this.updateEntries}
                                                                styles={{ root: { width: 70 } }}
                                                            />
                                                        </div>
                                                    </Stack>
                                                </div>
                                            </Stack>
                                        </div>
                                    </Stack>
                                    <SuccessTable trialIds={bestTrials.map(trial => trial.info.id)} />
                                </div>
                                <div className='box5'>
                                    <Command />
                                </div>
                                <div className='box6'>
                                    <Stack horizontal>
                                        <div style={itemStyle1}>
                                            <TitleContext.Provider
                                                value={{ text: 'Trial metric chart', icon: 'HomeGroup' }}
                                            >
                                                <Title />
                                            </TitleContext.Provider>
                                        </div>
                                        <div style={itemStyle2}>
                                            <Stack className='maxmin' horizontal>
                                                <div className='circle' />
                                                <div>{`Top ${this.context.metricGraphMode}imal trials`}</div>
                                            </Stack>
                                        </div>
                                    </Stack>
                                    <Accuracy accuracyData={accuracyGraphData} accNodata={noDataMessage} height={380} />
                                </div>
                            </div>
                        </div>
                    );
                }}
            </AppContext.Consumer>
        );
    }

    private findBestTrials(): Trial[] {
        const bestTrials = TRIALS.sort();
        const { bestTrialEntries, metricGraphMode } = this.context;
        if (metricGraphMode === 'max') {
            bestTrials.reverse().splice(JSON.parse(bestTrialEntries));
        } else {
            bestTrials.splice(JSON.parse(bestTrialEntries));
        }
        return bestTrials;
    }

    private generateAccuracyGraph(bestTrials: Trial[]): object {
        const xSequence = bestTrials.map(trial => trial.sequenceId);
        const ySequence = bestTrials.map(trial => trial.accuracy);

        return {
            // support max show 0.0000000
            grid: {
                left: 67,
                right: 40
            },
            tooltip: {
                trigger: 'item'
            },
            xAxis: {
                name: 'Trial',
                type: 'category',
                data: xSequence
            },
            yAxis: {
                name: 'Default metric',
                type: 'value',
                scale: true,
                data: ySequence
            },
            series: [
                {
                    symbolSize: 6,
                    type: 'scatter',
                    data: ySequence
                }
            ]
        };
    }
}

export default Overview;
