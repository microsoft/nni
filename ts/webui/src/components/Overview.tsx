import * as React from 'react';
import { Stack, Icon, Dropdown, DefaultButton } from '@fluentui/react';
import { EXPERIMENT, TRIALS } from '../static/datamodel';
import { Trial } from '../static/model/trial';
import { AppContext } from '../App';
import { Title } from './overview/Title';
import SuccessTable from './overview/table/SuccessTable';
import DefaultPoint from './trial-detail/DefaultMetricPoint';
import { BasicInfo } from './overview/params/BasicInfo';
import { ExpDuration } from './overview/count/ExpDuration';
import { ExpDurationContext } from './overview/count/ExpDurationContext';
import { TrialCount } from './overview/count/TrialCount';
import { Command1 } from './overview/command/Command1';
import { Command2 } from './overview/command/Command2';
import { TitleContext } from './overview/TitleContext';
import { itemStyleSucceed, entriesOption } from './overview/overviewConst';
import '../static/style/overview/overview.scss';
import '../static/style/overview/topTrial.scss';
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
        const maxExecDuration = EXPERIMENT.profile.params.maxExecDuration;
        const execDuration = EXPERIMENT.profile.execDuration;

        return (
            <AppContext.Consumer>
                {(value): React.ReactNode => {
                    const {
                        metricGraphMode,
                        bestTrialEntries,
                        maxDurationUnit,
                        expandRowIDs,
                        updateOverviewPage,
                        changeMaxDurationUnit,
                        changeExpandRowIDs
                    } = value;
                    const maxActive = metricGraphMode === 'max' ? 'active' : '';
                    const minActive = metricGraphMode === 'min' ? 'active' : '';
                    return (
                        <div className='overview'>
                            <div className='wrapper'>
                                {/* exp params */}
                                <div className='overviewBasicInfo'>
                                    <TitleContext.Provider value={{ text: 'Experiment', icon: 'AutoRacing' }}>
                                        <Title />
                                    </TitleContext.Provider>
                                    <BestMetricContext.Provider value={{ bestAccuracy: bestAccuracy }}>
                                        <BasicInfo />
                                    </BestMetricContext.Provider>
                                </div>
                                {/* duration & trial numbers */}
                                <div className='duration'>
                                    <TitleContext.Provider value={{ text: 'Duration', icon: 'Timer' }}>
                                        <Title />
                                    </TitleContext.Provider>
                                    <ExpDurationContext.Provider
                                        value={{
                                            maxExecDuration,
                                            execDuration,
                                            updateOverviewPage,
                                            maxDurationUnit,
                                            changeMaxDurationUnit
                                        }}
                                    >
                                        <ExpDuration />
                                    </ExpDurationContext.Provider>
                                </div>
                                <div className='trialCount'>
                                    <TitleContext.Provider value={{ text: 'Trial numbers', icon: 'NumberSymbol' }}>
                                        <Title />
                                    </TitleContext.Provider>
                                    <ExpDurationContext.Provider
                                        value={{
                                            maxExecDuration,
                                            execDuration,
                                            updateOverviewPage,
                                            maxDurationUnit,
                                            changeMaxDurationUnit
                                        }}
                                    >
                                        <TrialCount />
                                    </ExpDurationContext.Provider>
                                </div>
                                {/* table */}
                                <div className='overviewBestMetric'>
                                    <Stack horizontal>
                                        <div style={itemStyleSucceed}>
                                            <TitleContext.Provider value={{ text: 'Top trials', icon: 'BulletedList' }}>
                                                <Title />
                                            </TitleContext.Provider>
                                        </div>
                                        <div className='topTrialTitle'>
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
                                    <div className='overviewChart'>
                                        <DefaultPoint
                                            trialIds={bestTrials.map(trial => trial.info.trialJobId)}
                                            chartHeight={300}
                                            hasBestCurve={false}
                                            changeExpandRowIDs={changeExpandRowIDs}
                                        />
                                        <SuccessTable
                                            trialIds={bestTrials.map(trial => trial.info.trialJobId)}
                                            updateOverviewPage={updateOverviewPage}
                                            expandRowIDs={expandRowIDs}
                                            changeExpandRowIDs={changeExpandRowIDs}
                                        />
                                    </div>
                                </div>
                                <Stack className='overviewCommand' horizontal>
                                    <Command2 />
                                    <Command1 />
                                </Stack>
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

}

export default Overview;
