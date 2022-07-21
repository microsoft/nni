import * as React from 'react';
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

interface TrialDetailState {
    whichChart: string;
}

class TrialsDetail extends React.Component<{}, TrialDetailState> {
    static contextType = AppContext;
    context!: React.ContextType<typeof AppContext>;
    public interAccuracy = 0;
    public interAllTableList = 2;

    public tableList!: TableList | null;
    public searchInput!: HTMLInputElement | null;

    constructor(props) {
        super(props);
        this.state = {
            whichChart: 'Default metric'
        };
    }

    handleWhichTabs = (item: any): void => {
        this.setState({ whichChart: item.props.headerText });
    };

    render(): React.ReactNode {
        const { whichChart } = this.state;
        const source = TRIALS.toArray();
        const paraSource = TRIALS.succeededTrials();
        const succeededTrialIds = TRIALS.succeededTrials().map(trial => trial.id);
        return (
            <AppContext.Consumer>
                {(_value): React.ReactNode => (
                    <React.Fragment>
                        <div className='trial' id='tabsty'>
                            <Pivot
                                defaultSelectedKey={'0'}
                                className='detail-title'
                                onLinkClick={this.handleWhichTabs}
                                selectedKey={whichChart}
                            >
                                {/* <PivotItem tab={this.titleOfacc} key="1"> doesn't work*/}
                                <PivotItem headerText='Default metric' itemIcon='HomeGroup' key='Default metric'>
                                    <Stack className='graph'>
                                        <DefaultPoint
                                            trialIds={succeededTrialIds}
                                            hasBestCurve={true}
                                            chartHeight={402}
                                            changeExpandRowIDs={_value.changeExpandRowIDs}
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
                                    <Duration source={TRIALS.unWaittingTrials()} />
                                </PivotItem>
                                {/* <PivotItem tab={this.titleOfIntermediate} key="4"> */}
                                <PivotItem
                                    headerText='Intermediate result'
                                    itemIcon='StackedLineChart'
                                    key='Intermediate result'
                                >
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
                )}
            </AppContext.Consumer>
        );
    }
}

export default TrialsDetail;
