import React, { useContext } from 'react';
import { Stack, Pivot, PivotItem } from '@fluentui/react';
import { EXPERIMENT, TRIALS } from '@static/datamodel';
import { AppContext } from '@/App';
import DefaultPoint from './chart/DefaultMetricPoint';
import Duration from './chart/Duration';
import Para from './chart/Para';
import Intermediate from './chart/Intermediate';
import TableList from './table/TableList';
import '@style/button.scss';
import '@style/openRow.scss';
import '@style/pagination.scss';
import '@style/experiment/overview/overviewTitle.scss';
import '@style/experiment/trialdetail/search.scss';
import '@style/experiment/trialdetail/tensorboard.scss';
import '@style/table.scss';
import '@style/common/trialStatus.css';

/**
 * single experiment
 * trial detail page
 */

const TrialsDetail = (): any => {
    const { changeExpandRowIDs } = useContext(AppContext);
    const source = TRIALS.toArray();
    const paraSource = TRIALS.succeededTrials();
    const succeededTrialIds = TRIALS.succeededTrials().map(trial => trial.id);
    return (
        <React.Fragment>
            <div className='trial' id='tabsty'>
                <Pivot className='detail-title'>
                    <PivotItem headerText='Default metric' itemIcon='HomeGroup'>
                        <Stack className='graph'>
                            <DefaultPoint
                                trialIds={succeededTrialIds}
                                hasBestCurve={true}
                                chartHeight={402}
                                changeExpandRowIDs={changeExpandRowIDs}
                            />
                        </Stack>
                    </PivotItem>
                    <PivotItem headerText='Hyper-parameter' itemIcon='Equalizer'>
                        <Stack className='graph'>
                            <Para trials={paraSource} searchSpace={EXPERIMENT.searchSpaceNew} />
                        </Stack>
                    </PivotItem>
                    <PivotItem headerText='Duration' itemIcon='BarChartHorizontal'>
                        <Duration source={TRIALS.notWaittingTrials()} />
                    </PivotItem>
                    <PivotItem headerText='Intermediate result' itemIcon='StackedLineChart'>
                        {/* why this graph has small footprint? */}
                        <Intermediate source={TRIALS.allTrialsIntermediateChart()} />
                    </PivotItem>
                </Pivot>
            </div>
            {/* trial table list */}
            <div className='detailTable' style={{ marginTop: 10 }}>
                <TableList tableSource={source} />
            </div>
        </React.Fragment>
    );
};

export default TrialsDetail;
