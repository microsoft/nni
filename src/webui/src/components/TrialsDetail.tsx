import * as React from 'react';
import {
    Stack, Pivot, PivotItem, StackItem, PrimaryButton, Dropdown, IDropdownOption
} from 'office-ui-fabric-react';
import { EXPERIMENT, TRIALS } from '../static/datamodel';
import { Trial } from '../static/model/trial'; // eslint-disable-line no-unused-vars
import DefaultPoint from './trial-detail/DefaultMetricPoint';
import Duration from './trial-detail/Duration';
import Title1 from './overview/Title1';
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
            whichGraph: '1',
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
            case 'id':
                filter = (trial): boolean => trial.info.id.toUpperCase().includes(targetValue.toUpperCase());
                break;
            case 'Trial No.':
                filter = (trial): boolean => trial.info.sequenceId.toString() === targetValue;
                break;
            case 'Status':
                filter = (trial): boolean => trial.info.status.toUpperCase().includes(targetValue.toUpperCase());
                break;
            case 'parameters':
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
            console.info(item.text); // eslint-disable-line
            this.setState({ tablePageSize: item.text === 'all' ? -1 : parseInt(item.text, 10) });
        }
    }

    handleWhichTabs = (activeKey: string): void => {
        this.setState({ whichGraph: activeKey });
    }

    updateSearchFilterType = (event: React.FormEvent<HTMLDivElement>, item: IDropdownOption | undefined): void => {
        // clear input value and re-render table
        if (item !== undefined) {
            if (this.searchInput !== null) {
                this.searchInput.value = '';
            }
            // console.info(item.text); // eslint-disable-line
            this.setState(() => ({ searchType: item.text }));
        }
    }

    render(): React.ReactNode {
        const { tablePageSize, whichGraph, searchType } = this.state;
        const { columnList, changeColumn } = this.props;
        const source = TRIALS.filter(this.state.searchFilter);
        const trialIds = TRIALS.filter(this.state.searchFilter).map(trial => trial.id);
        const options = [
            { key: '20', text: '20' },
            { key: '50', text: '50' },
            { key: '100', text: '100' },
            { key: 'all', text: 'all' },
        ];
        const searchOptions = [
            { key: 'Id', text: 'Id' },
            { key: 'Trial No.', text: 'Trial No.' },
            { key: 'Status', text: 'Status' },
            { key: 'Parameters', text: 'Parameters' },
        ];
        return (
            <div>
                <div className="trial" id="tabsty">
                    <Pivot defaultSelectedKey={"0"}>
                        {/* <PivotItem tab={this.titleOfacc} key="1"> doesn't work*/}
                        <PivotItem headerText="Default metric" itemIcon="Recent" key="1">
                            <Stack className="graph">
                                <DefaultPoint
                                    trialIds={trialIds}
                                    visible={whichGraph === '1'}
                                    trialsUpdateBroadcast={this.props.trialsUpdateBroadcast}
                                />
                            </Stack>
                        </PivotItem>
                        {/* <PivotItem tab={this.titleOfhyper} key="2"> */}
                        <PivotItem headerText="Hyper-parameter" itemIcon="Recent" key="2">
                            <Stack className="graph">
                                <Para
                                    dataSource={source}
                                    expSearchSpace={JSON.stringify(EXPERIMENT.searchSpace)}
                                    whichGraph={whichGraph}
                                />
                            </Stack>
                        </PivotItem>
                        {/* <PivotItem tab={this.titleOfDuration} key="3"> */}
                        <PivotItem headerText="Duration" itemIcon="Recent" key="3">
                            <Duration source={source} whichGraph={whichGraph} />
                        </PivotItem>
                        {/* <PivotItem tab={this.titleOfIntermediate} key="4"> */}
                        <PivotItem headerText="Intermediate result" itemIcon="Recent" key="4">
                            <div className="graphContent">
                                <Intermediate source={source} whichGraph={whichGraph} />
                            </div>
                        </PivotItem>
                    </Pivot>
                </div>
                {/* trial table list */}
                <Title1 text="Trial jobs" icon="6.png" />
                <Stack horizontal className="allList">
                    <StackItem grow={50}>
                        <Stack horizontal>
                            <span>Show</span>
                            <Dropdown
                                selectedKey={tablePageSize ? tablePageSize.toString() : undefined}
                                defaultSelectedKeys={['20']}
                                options={options}
                                onChange={this.handleTablePageSizeSelect}
                                styles={{root: {width: 80}}}
                            />
                            <span>entries</span>
                        </Stack>
                    </StackItem>
                    <StackItem grow={50}>
                        <Stack horizontal horizontalAlign="end">
                            <PrimaryButton
                                onClick={(): void => { if (this.tableList) { this.tableList.addColumn(); } }}
                            >
                                Add column
                            </PrimaryButton>
                            <PrimaryButton
                                // use child-component tableList's function, the function is in child-component.
                                onClick={(): void => { if (this.tableList) { this.tableList.compareBtn(); } }}
                            >
                                Compare
                            </PrimaryButton>
                            <Dropdown
                                selectedKey={searchType}
                                options={searchOptions}
                                onChange={this.updateSearchFilterType}
                                styles={{root: {width: 150}}}
                            />
                            <input
                                type="text"
                                className="search-input"
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
        );
    }
}

export default TrialsDetail;
