import React, { useContext } from 'react';
import { Stack, Dropdown, DefaultButton } from '@fluentui/react';
import { TRIALS } from '@static/datamodel';
import { Trial } from '@static/model/trial';
import { AppContext } from '@/App';
import SuccessTable from './table/SuccessTable';
import DefaultPoint from '../trialdetail/chart/DefaultMetricPoint';
import { BasicInfo } from './basic/BasicInfo';
import { Duration } from './count/Duration';
import { TrialCount } from './count/TrialNumbers';
import { buttonsGap } from '@components/common/Gap';
import '@style/experiment/overview/overview.scss';
import '@style/experiment/overview/topTrial.scss';
import '@style/table.scss';

const entriesOption = [
    { key: '10', text: '10' },
    { key: '20', text: '20' },
    { key: '30', text: '30' },
    { key: '50', text: '50' },
    { key: '100', text: '100' }
];
/**
 * single experiment
 * overview page
 */

export const BestMetricContext = React.createContext({
    bestAccuracy: 0
});

const Overview = (): any => {
    const { changeMetricGraphMode, changeEntries } = useContext(AppContext); // global

    const clickMaxTop = (event: React.SyntheticEvent<EventTarget>): void => {
        event.stopPropagation();
        // #999 panel active bgcolor; #b3b3b3 as usual
        changeMetricGraphMode('Maximize');
    };

    const clickMinTop = (event: React.SyntheticEvent<EventTarget>): void => {
        event.stopPropagation();
        changeMetricGraphMode('Minimize');
    };

    const updateEntries = (event: React.FormEvent<HTMLDivElement>, item: any): void => {
        if (item !== undefined) {
            changeEntries(item.key);
        }
    };

    const findBestTrials = (): Trial[] => {
        const bestTrials = TRIALS.sort();
        const { bestTrialEntries, metricGraphMode } = useContext(AppContext);
        if (metricGraphMode === 'Maximize') {
            bestTrials.reverse().splice(JSON.parse(bestTrialEntries));
        } else {
            bestTrials.splice(JSON.parse(bestTrialEntries));
        }
        return bestTrials;
    };

    const bestTrials = findBestTrials();
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    const bestAccuracy = bestTrials.length > 0 ? bestTrials[0].accuracy! : NaN;
    const { metricGraphMode, bestTrialEntries, expandRowIDs, updateOverviewPage, changeExpandRowIDs } =
        useContext(AppContext);
    return (
        <div className='overview'>
            <div className='overviewWrapper'>
                {/* exp params */}
                <div className='overviewBasicInfo left'>
                    {/* Provider is required */}
                    <BestMetricContext.Provider value={{ bestAccuracy: bestAccuracy }}>
                        <BasicInfo />
                    </BestMetricContext.Provider>
                </div>
                {/* duration & trial numbers */}
                <div className='duration left'>
                    <h3 className='title'>Duration</h3>
                    <Duration />
                </div>
                <div className='trialCount left'>
                    <h3 className='title'>Trial numbers</h3>
                    <TrialCount />
                </div>
                {/* table */}
                <div className='overviewBestMetric'>
                    <Stack horizontal horizontalAlign='space-between'>
                        <div>
                            <h3 className='title'>
                                Top trials<span className='font-untheme'>{bestTrialEntries}</span>
                            </h3>
                        </div>
                        <Stack horizontal horizontalAlign='end' tokens={buttonsGap}>
                            <DefaultButton
                                text='Maximize'
                                className={metricGraphMode === 'Maximize' ? 'active' : ''}
                                styles={{ root: { padding: '0 8px' } }}
                                iconProps={{ iconName: 'Market' }}
                                onClick={clickMaxTop}
                            />
                            <DefaultButton
                                text='Minimize'
                                className={`mincenter ${metricGraphMode === 'Minimize' ? 'active' : ''}`}
                                styles={{ root: { padding: '0 8px' } }}
                                iconProps={{ iconName: 'MarketDown' }}
                                onClick={clickMinTop}
                            />
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
                    </Stack>
                    <div className='overviewChart'>
                        <DefaultPoint
                            trialIds={bestTrials.map(trial => trial.info.trialJobId)}
                            chartHeight={530}
                            hasBestCurve={false}
                            changeExpandRowIDs={changeExpandRowIDs}
                        />
                    </div>
                </div>
                <div className='bestTable'>
                    <SuccessTable
                        trialIds={bestTrials.map(trial => trial.info.trialJobId)}
                        updateOverviewPage={updateOverviewPage}
                        expandRowIDs={expandRowIDs}
                        changeExpandRowIDs={changeExpandRowIDs}
                    />
                </div>
            </div>
        </div>
    );
};

export default Overview;
