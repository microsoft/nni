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
    whichGraph: string;
    searchType: string;
    searchFilter: (trial: Trial) => boolean;
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
            // whichGraph: '1',
            whichGraph: 'Default metric',
            searchType: 'Id',
            // eslint-disable-next-line @typescript-eslint/no-unused-vars, @typescript-eslint/explicit-function-return-type
            searchFilter: trial => true
        };
    }

    // search a trial by trial No. | trial id | Parameters | Status
    searchTrial = (event: React.ChangeEvent<HTMLInputElement>): void => {
        const targetValue = event.target.value;
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        let filter = (trial: Trial): boolean => true;
        if (!targetValue.trim()) {
            this.setState({ searchFilter: filter });
            return;
        }
        switch (this.state.searchType) {
            case 'Id':
                filter = (trial): boolean => trial.info.id.toUpperCase().includes(targetValue.toUpperCase());
                break;
            case 'Trial No.':
                filter = (trial): boolean => trial.info.sequenceId.toString() === targetValue;
                break;
            case 'Status':
                filter = (trial): boolean => trial.info.status.toUpperCase().includes(targetValue.toUpperCase());
                break;
            case 'Parameters':
                // TODO: support filters like `x: 2` (instead of `"x": 2`)
                filter = (trial): boolean => JSON.stringify(trial.info.hyperParameters, null, 4).includes(targetValue);
                break;
            default:
                alert(`Unexpected search filter ${this.state.searchType}`);
        }
        this.setState({ searchFilter: filter });
    }

    handleTablePageSizeSelect = (event: React.FormEvent<HTMLDivElement>, item: IDropdownOption | undefined): void => {
        if (item !== undefined) {
            this.setState({ tablePageSize: item.text === 'all' ? -1 : parseInt(item.text, 10) });
        }
    }

    handleWhichTabs = (item: any): void => {
        this.setState({whichGraph: item.props.headerText});
    }

    updateSearchFilterType = (event: React.FormEvent<HTMLDivElement>, item: IDropdownOption | undefined): void => {
        // clear input value and re-render table
        if (item !== undefined) {
            if (this.searchInput !== null) {
                this.searchInput.value = '';
            }
            this.setState(() => ({ searchType: item.text }));
        }
    }

    render(): React.ReactNode {
        const { tablePageSize, whichGraph, searchType } = this.state;
        const { columnList, changeColumn } = this.props;
        const source = TRIALS.filter(this.state.searchFilter);
        const trialIds = TRIALS.filter(this.state.searchFilter).map(trial => trial.id);
        const searchOptions = [
            { key: 'Id', text: 'Id' },
            { key: 'Trial No.', text: 'Trial No.' },
            { key: 'Status', text: 'Status' },
            { key: 'Parameters', text: 'Parameters' },
        ];
        return (
            <div>
                <div className="trial" id="tabsty">
                    <Pivot defaultSelectedKey={"0"} className="detial-title" onLinkClick={this.handleWhichTabs} selectedKey={whichGraph}>
                        {/* <PivotItem tab={this.titleOfacc} key="1"> doesn't work*/}
                        <PivotItem headerText="Default metric" itemIcon="HomeGroup" key="Default metric">
                            <Stack className="graph">
                                <DefaultPoint
                                    trialIds={trialIds}
                                    visible={whichGraph === 'Default metric'}
                                    trialsUpdateBroadcast={this.props.trialsUpdateBroadcast}
                                />
                            </Stack>
                        </PivotItem>
                        {/* <PivotItem tab={this.titleOfhyper} key="2"> */}
                        <PivotItem headerText="Hyper-parameter" itemIcon="Equalizer" key="Hyper-parameter">
                            <Stack className="graph">
                                <Para
                                    dataSource={source}
                                    expSearchSpace={JSON.stringify(EXPERIMENT.searchSpace)}
                                    whichGraph={whichGraph}
                                />
                            </Stack>
                        </PivotItem>
                        {/* <PivotItem tab={this.titleOfDuration} key="3"> */}
                        <PivotItem headerText="Duration" itemIcon="BarChartHorizontal" key="Duration">
                            <Duration source={source} whichGraph={whichGraph} />
                        </PivotItem>
                        {/* <PivotItem tab={this.titleOfIntermediate} key="4"> */}
                        <PivotItem headerText="Intermediate result" itemIcon="StackedLineChart" key="Intermediate result">
                            {/* *why this graph has small footprint? */}
                            <Intermediate source={source} whichGraph={whichGraph} />
                        </PivotItem>
                    </Pivot>
                </div>
                {/* trial table list */}
                <div style={{ backgroundColor: '#fff' }}>
                    <Stack horizontal className="panelTitle" style={{ marginTop: 10 }}>
                        <span style={{ marginRight: 12 }}>{tableListIcon}</span>
                        <span>Trial jobs</span>
                    </Stack>
                    <Stack horizontal className="allList">
                        <StackItem grow={50}>
                            <DefaultButton
                                text="Compare"
                                className="allList-compare"
                                // use child-component tableList's function, the function is in child-component.
                                onClick={(): void => { if (this.tableList) { this.tableList.compareBtn(); } }}
                            />
                        </StackItem>
                        <StackItem grow={50}>
                            <Stack horizontal horizontalAlign="end" className="allList">
                                <DefaultButton
                                    className="allList-button-gap"
                                    text="Add column"
                                    onClick={(): void => { if (this.tableList) { this.tableList.addColumn(); } }}
                                />
                                <Dropdown
                                    selectedKey={searchType}
                                    options={searchOptions}
                                    onChange={this.updateSearchFilterType}
                                    styles={{ root: { width: 150 } }}
                                />
                                <input
                                    type="text"
                                    className="allList-search-input"
                                    placeholder={`Search by ${this.state.searchType}`}
                                    onChange={this.searchTrial}
                                    style={{ width: 230 }}
                                    ref={(text): any => (this.searchInput) = text}
                                />
                            </Stack>
                        </StackItem>
                    </Stack>
                    <TableList
                        pageSize={tablePageSize}
                        tableSource={source.map(trial => trial.tableRecord)}
                        columnList={columnList}
                        changeColumn={changeColumn}
                        trialsUpdateBroadcast={this.props.trialsUpdateBroadcast}
                        // TODO: change any to specific type
                        ref={(tabList): any => this.tableList = tabList}
                    />
                </div>
            </div>
        );
    }
}

export default TrialsDetail;
