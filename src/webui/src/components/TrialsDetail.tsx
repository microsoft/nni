import * as React from 'react';
import { Stack, StackItem, Pivot, PivotItem, Dropdown, IDropdownOption, DefaultButton } from '@fluentui/react';
import { EXPERIMENT, TRIALS } from '../static/datamodel';
import { Trial } from '../static/model/trial';
import { AppContext } from '../App';
import { tableListIcon } from './buttons/Icon';
import DefaultPoint from './trial-detail/DefaultMetricPoint';
import Duration from './trial-detail/Duration';
import Para from './trial-detail/Para';
import Intermediate from './trial-detail/Intermediate';
import TableList from './trial-detail/TableList';
import '../static/style/trialsDetail.scss';
import '../static/style/search.scss';

const searchOptions = [
    { key: 'id', text: 'Id' },
    { key: 'Trial No.', text: 'Trial No.' },
    { key: 'status', text: 'Status' },
    { key: 'parameters', text: 'Parameters' }
];

interface TrialDetailState {
    tablePageSize: number; // table components val
    whichChart: string;
}

class TrialsDetail extends React.Component<{}, TrialDetailState> {
    static contextType = AppContext;
    public interAccuracy = 0;
    public interAllTableList = 2;

    public tableList!: TableList | null;
    public searchInput!: HTMLInputElement | null;

    constructor(props) {
        super(props);
        this.state = {
            tablePageSize: 20,
            whichChart: 'Default metric',
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
            case 'status':
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
    };

    handleTablePageSizeSelect = (event: React.FormEvent<HTMLDivElement>, item: IDropdownOption | undefined): void => {
        if (item !== undefined) {
            this.setState({ tablePageSize: item.text === 'all' ? -1 : parseInt(item.text, 10) });
        }
    };

    handleWhichTabs = (item: any): void => {
        this.setState({ whichChart: item.props.headerText });
    };

    updateSearchFilterType = (event: React.FormEvent<HTMLDivElement>, item: IDropdownOption | undefined): void => {
        // clear input value and re-render table
        if (item !== undefined) {
            if (this.searchInput !== null) {
                this.searchInput.value = '';
            }
            this.setState(() => ({ searchType: item.key.toString() }));
        }
    };

    render(): React.ReactNode {
        const { tablePageSize, whichChart, searchType } = this.state;
        const source = TRIALS.filter(this.state.searchFilter);
        const trialIds = TRIALS.filter(this.state.searchFilter).map(trial => trial.id);

        return (
            <AppContext.Consumer>
                {(value): React.ReactNode => (
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
                        <div style={{ backgroundColor: '#fff' }}>
                            <Stack horizontal className='panelTitle' style={{ marginTop: 10 }}>
                                <span style={{ marginRight: 12 }}>{tableListIcon}</span>
                                <span>Trial jobs</span>
                            </Stack>
                            <Stack horizontal className='allList'>
                                <StackItem grow={50}>
                                    <DefaultButton
                                        text='Compare'
                                        className='allList-compare'
                                        // use child-component tableList's function, the function is in child-component.
                                        onClick={(): void => {
                                            if (this.tableList) {
                                                this.tableList.compareBtn();
                                            }
                                        }}
                                    />
                                </StackItem>
                                <StackItem grow={50}>
                                    <Stack horizontal horizontalAlign='end' className='allList'>
                                        <DefaultButton
                                            className='allList-button-gap'
                                            text='Add column'
                                            onClick={(): void => {
                                                if (this.tableList) {
                                                    this.tableList.addColumn();
                                                }
                                            }}
                                        />
                                        <Dropdown
                                            selectedKey={searchType}
                                            options={searchOptions}
                                            onChange={this.updateSearchFilterType}
                                            styles={{ root: { width: 150 } }}
                                        />
                                        <input
                                            type='text'
                                            className='allList-search-input'
                                            placeholder={`Search by ${this.state.searchType}`}
                                            onChange={this.searchTrial}
                                            style={{ width: 230 }}
                                            ref={(text): any => (this.searchInput = text)}
                                        />
                                    </Stack>
                                </StackItem>
                            </Stack>
                            <TableList
                                pageSize={tablePageSize}
                                tableSource={source.map(trial => trial.tableRecord)}
                                columnList={value.columnList}
                                changeColumn={value.changeColumn}
                                trialsUpdateBroadcast={this.context.trialsUpdateBroadcast}
                                // TODO: change any to specific type
                                ref={(tabList): any => (this.tableList = tabList)}
                            />
                        </div>
                    </React.Fragment>
                )}
            </AppContext.Consumer>
        );
    }
}

export default TrialsDetail;
