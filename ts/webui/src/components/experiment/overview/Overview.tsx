import React, { useContext } from 'react';
import { Stack, Icon, Dropdown, DefaultButton } from '@fluentui/react';
import { EXPERIMENT, TRIALS } from '@static/datamodel';
import { Trial } from '@static/model/trial';
import { AppContext } from '@/App';
import { Title } from './Title';
import SuccessTable from './table/SuccessTable';
import DefaultPoint from '../trialdetail/chart/DefaultMetricPoint';
import { BasicInfo } from './params/BasicInfo';
import { ExpDuration } from './count/ExpDuration';
import { TrialCount } from './count/TrialCount';
import Config from './Config';
import { TitleContext } from './TitleContext';
import { itemStyleSucceed, entriesOption } from './overviewConst';
import '@style/experiment/overview/overview.scss';
import '@style/experiment/overview/topTrial.scss';
import '@style/table.scss';

/**
 * single experiment
 * overview page
 */

export const BestMetricContext = React.createContext({
    bestAccuracy: 0
});

const Overview = (): any => {
    const clickMaxTop = (event: React.SyntheticEvent<EventTarget>): void => {
        event.stopPropagation();
        // #999 panel active bgcolor; #b3b3b3 as usual
        const { changeMetricGraphMode } = useContext(AppContext);
        changeMetricGraphMode('max');
    };

    const clickMinTop = (event: React.SyntheticEvent<EventTarget>): void => {
        event.stopPropagation();
        const { changeMetricGraphMode } = useContext(AppContext);
        changeMetricGraphMode('min');
    };

    const updateEntries = (event: React.FormEvent<HTMLDivElement>, item: any): void => {
        if (item !== undefined) {
            const { changeEntries } = useContext(AppContext);
            changeEntries(item.key);
        }
    };

    const findBestTrials = (): Trial[] => {
        const bestTrials = TRIALS.sort();
        const { bestTrialEntries, metricGraphMode } = useContext(AppContext);
        if (metricGraphMode === 'max') {
            bestTrials.reverse().splice(JSON.parse(bestTrialEntries));
        } else {
            bestTrials.splice(JSON.parse(bestTrialEntries));
        }
        return bestTrials;
    }

    const bestTrials = findBestTrials();
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const bestAccuracy = bestTrials.length > 0 ? bestTrials[0].accuracy! : NaN;
    const { metricGraphMode,
        bestTrialEntries,
        expandRowIDs,
        updateOverviewPage,
        changeExpandRowIDs } = useContext(AppContext);
    return (
        <div className='overview'>
            <div className='overviewWrapper'>
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
                    <ExpDuration />
                </div>
                <div className='trialCount'>
                    <TitleContext.Provider value={{ text: 'Trial numbers', icon: 'NumberSymbol' }}>
                        <Title />
                    </TitleContext.Provider>
                    <TrialCount />
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
                                    onClick={clickMaxTop}
                                    className={metricGraphMode === 'max' ? 'active' : ''}
                                    styles={{ root: { minWidth: 70, padding: 0 } }}
                                >
                                    <Icon iconName='Market' />
                                    <span className='max'>Max</span>
                                </DefaultButton>
                                <div className='mincenter'>
                                    <DefaultButton
                                        onClick={clickMinTop}
                                        className={metricGraphMode === 'min' ? 'active' : ''}
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
                                                onChange={updateEntries}
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
                <Stack className='overviewCommand'>
                    <Config />
                </Stack>
            </div>
        </div>
    );
};

export default Overview;
