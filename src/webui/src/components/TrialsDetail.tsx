import * as React from 'react';
import {
    Stack, StackItem, Pivot, PivotItem, Dropdown, IDropdownOption, DefaultButton
} from 'office-ui-fabric-react';
import { EXPERIMENT, TRIALS } from '../static/datamodel';
import { Trial } from '../static/model/trial';
import { tableListIcon } from './Buttons/Icon';
import DefaultPoint from './trial-detail/DefaultMetricPoint';
import Duration from './trial-detail/Duration';
import Para from './trial-detail/Para';
import Intermediate from './trial-detail/Intermediate';
import TableList from './trial-detail/TableList';
import '../static/style/trialsDetail.scss';
import '../static/style/search.scss';

interface TrialDetailState {
    tablePageSize: number; // table components val
    whichChart: string;
}

interface TrialsDetailProps {
    columnList: string[];
    changeColumn: (val: string[]) => void;
    experimentUpdateBroacast: number;
    trialsUpdateBroadcast: number;
}

class TrialsDetail extends React.Component<TrialsDetailProps, TrialDetailState> {

    public interAccuracy = 0;
    public interAllTableList = 2;

    public tableList!: TableList | null;
    public searchInput!: HTMLInputElement | null;

    constructor(props: TrialsDetailProps) {
        super(props);
        this.state = {
            tablePageSize: 20,
            whichChart: 'Default metric',
        };
    }

    handleTablePageSizeSelect = (event: React.FormEvent<HTMLDivElement>, item: IDropdownOption | undefined): void => {
        if (item !== undefined) {
            this.setState({ tablePageSize: item.text === 'all' ? -1 : parseInt(item.text, 10) });
        }
    }

    handleWhichTabs = (item: any): void => {
        this.setState({whichChart: item.props.headerText});
    }

    updateSearchFilterType = (event: React.FormEvent<HTMLDivElement>, item: IDropdownOption | undefined): void => {
        // clear input value and re-render table
        if (item !== undefined) {
            if (this.searchInput !== null) {
                this.searchInput.value = '';
            }
            this.setState(() => ({ searchType: item.key.toString() }));
        }
    }

    render(): React.ReactNode {
        const { tablePageSize, whichChart, searchType } = this.state;
        const { columnList, changeColumn, trialsUpdateBroadcast } = this.props;
        const source = TRIALS.filter(this.state.searchFilter);
        const trialIds = TRIALS.filter(this.state.searchFilter).map(trial => trial.id);
        const searchOptions = [
            { key: 'id', text: 'Id' },
            { key: 'Trial No.', text: 'Trial No.' },
            { key: 'status', text: 'Status' },
            { key: 'parameters', text: 'Parameters' },
        ];
        return (
            <div>
                <div className="trial" id="tabsty">
                    <Pivot defaultSelectedKey={"0"} className="detial-title" onLinkClick={this.handleWhichTabs} selectedKey={whichChart}>
                        {/* <PivotItem tab={this.titleOfacc} key="1"> doesn't work*/}
                        <PivotItem headerText="Default metric" itemIcon="HomeGroup" key="Default metric">
                            <Stack className="graph">
                                <DefaultPoint
                                    trialIds={trialIds}
                                    visible={whichChart === 'Default metric'}
                                    trialsUpdateBroadcast={trialsUpdateBroadcast}
                                />
                            </Stack>
                        </PivotItem>
                        {/* <PivotItem tab={this.titleOfhyper} key="2"> */}
                        <PivotItem headerText="Hyper-parameter" itemIcon="Equalizer" key="Hyper-parameter">
                            <Stack className="graph">
                                <Para
                                    trials={source}
                                    searchSpace={EXPERIMENT.searchSpaceNew}
                                    whichChart={whichChart}
                                />
                            </Stack>
                        </PivotItem>
                        {/* <PivotItem tab={this.titleOfDuration} key="3"> */}
                        <PivotItem headerText="Duration" itemIcon="BarChartHorizontal" key="Duration">
                            <Duration source={source} whichChart={whichChart} />
                        </PivotItem>
                        {/* <PivotItem tab={this.titleOfIntermediate} key="4"> */}
                        <PivotItem headerText="Intermediate result" itemIcon="StackedLineChart" key="Intermediate result">
                            {/* *why this graph has small footprint? */}
                            <Intermediate source={source} whichChart={whichChart} />
                        </PivotItem>
                    </Pivot>
                </div>
                {/* trial table list */}
                {/* in this version, it will be a separate component with a completely indepedent filter */}
                <TableList
                    trialsUpdateBroadcast={trialsUpdateBroadcast}
                    tableSource={source}
                />
            </div>
        );
    }
}

export default TrialsDetail;
