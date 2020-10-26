import * as React from 'react';
import { Stack, Pivot, PivotItem } from '@fluentui/react';
import { EXPERIMENT, TRIALS } from '../static/datamodel';
import { AppContext } from '../App';
import DefaultPoint from './trial-detail/DefaultMetricPoint';
import Duration from './trial-detail/Duration';
import Para from './trial-detail/Para';
import Intermediate from './trial-detail/Intermediate';
import TableList from './trial-detail/TableList';
import '../static/style/trialsDetail.scss';
import '../static/style/search.scss';

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
        const trialIds = TRIALS.toArray().map(trial => trial.id);

        return (
            <AppContext.Consumer>
                {(_value): React.ReactNode => (
                    <React.Fragment>
                        <div className='trial' id='tabsty'>
                            <Pivot
                                defaultSelectedKey={'0'}
                                className='detial-title'
                                onLinkClick={this.handleWhichTabs}
                                selectedKey={whichChart}
                            >
                                {/* <PivotItem tab={this.titleOfacc} key="1"> doesn't work*/}
                                <PivotItem headerText='Default metric' itemIcon='HomeGroup' key='Default metric'>
                                    <Stack className='graph'>
                                        <DefaultPoint trialIds={trialIds} visible={whichChart === 'Default metric'} />
                                    </Stack>
                                </PivotItem>
                                {/* <PivotItem tab={this.titleOfhyper} key="2"> */}
                                <PivotItem headerText='Hyper-parameter' itemIcon='Equalizer' key='Hyper-parameter'>
                                    <Stack className='graph'>
                                        <Para
                                            trials={source}
                                            searchSpace={EXPERIMENT.searchSpaceNew}
                                            whichChart={whichChart}
                                        />
                                    </Stack>
                                </PivotItem>
                                {/* <PivotItem tab={this.titleOfDuration} key="3"> */}
                                <PivotItem headerText='Duration' itemIcon='BarChartHorizontal' key='Duration'>
                                    <Duration source={source} whichChart={whichChart} />
                                </PivotItem>
                                {/* <PivotItem tab={this.titleOfIntermediate} key="4"> */}
                                <PivotItem
                                    headerText='Intermediate result'
                                    itemIcon='StackedLineChart'
                                    key='Intermediate result'
                                >
                                    {/* *why this graph has small footprint? */}
                                    <Intermediate source={source} whichChart={whichChart} />
                                </PivotItem>
                            </Pivot>
                        </div>
                        {/* trial table list */}
                        <div style={{ backgroundColor: '#fff', marginTop: 10 }}>
                            <TableList
                                tableSource={source}
                                trialsUpdateBroadcast={this.context.trialsUpdateBroadcast}
                            />
                        </div>
                    </React.Fragment>
                )}
            </AppContext.Consumer>
        );
    }
}

export default TrialsDetail;
