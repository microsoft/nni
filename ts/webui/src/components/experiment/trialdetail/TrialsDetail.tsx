import React, { useState, useContext } from 'react';
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
    const [whichChart, setChart] = useState('Default metric' as string);
    const handleWhichTabs = (item: any): void => {
        setChart(item.props.headerText);
    };
    const source = TRIALS.toArray();
    const paraSource = TRIALS.succeededTrials();
    const succeededTrialIds = TRIALS.succeededTrials().map(trial => trial.id);
    return (
        <React.Fragment>
            <div className='trial' id='tabsty'>
                <Pivot
                    defaultSelectedKey={'0'}
                    className='detail-title'
                    onLinkClick={handleWhichTabs}
                    selectedKey={whichChart}
                >
                    {/* <PivotItem tab={this.titleOfacc} key="1"> doesn't work*/}
                    <PivotItem headerText='Default metric' itemIcon='HomeGroup' key='Default metric'>
                        <Stack className='graph'>
                            <DefaultPoint
                                trialIds={succeededTrialIds}
                                hasBestCurve={true}
                                chartHeight={402}
                                changeExpandRowIDs={changeExpandRowIDs}
                            />
                        </Stack>
                    </PivotItem>
                    {/* <PivotItem tab={this.titleOfhyper} key="2"> */}
                    <PivotItem headerText='Hyper-parameter' itemIcon='Equalizer' key='Hyper-parameter'>
                        <Stack className='graph'>
                            <Para trials={paraSource} searchSpace={EXPERIMENT.searchSpaceNew} />
                        </Stack>
                    </PivotItem>
                    {/* <PivotItem tab={this.titleOfDuration} key="3"> */}
                    <PivotItem headerText='Duration' itemIcon='BarChartHorizontal' key='Duration'>
                        <Duration source={TRIALS.notWaittingTrials()} />
                    </PivotItem>
                    {/* <PivotItem tab={this.titleOfIntermediate} key="4"> */}
                    <PivotItem headerText='Intermediate result' itemIcon='StackedLineChart' key='Intermediate result'>
                        {/* *why this graph has small footprint? */}
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
