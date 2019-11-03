import * as React from 'react';
import { Row, Col, Tabs, Select, Button, Icon } from 'antd';
const Option = Select.Option;
import { EXPERIMENT, TRIALS } from '../static/datamodel';
import { Trial } from '../static/model/trial';
import DefaultPoint from './trial-detail/DefaultMetricPoint';
import Duration from './trial-detail/Duration';
import Title1 from './overview/Title1';
import Para from './trial-detail/Para';
import Intermediate from './trial-detail/Intermediate';
import TableList from './trial-detail/TableList';
const TabPane = Tabs.TabPane;
import '../static/style/trialsDetail.scss';
import '../static/style/search.scss';

interface TrialDetailState {
    tablePageSize: number; // table components val
    whichGraph: string;
    searchType: string;
    searchFilter: (trial: Trial) => boolean;
}

interface TrialsDetailProps {
    columnList: Array<string>;
    changeColumn: (val: Array<string>) => void;
    experimentUpdateBroacast: number;
    trialsUpdateBroadcast: number;
}

class TrialsDetail extends React.Component<TrialsDetailProps, TrialDetailState> {

    public interAccuracy = 0;
    public interAllTableList = 2;

    public tableList: TableList | null;
    public searchInput: HTMLInputElement | null;

    private titleOfacc = (
        <Title1 text="Default metric" icon="3.png" />
    );

    private titleOfhyper = (
        <Title1 text="Hyper-parameter" icon="1.png" />
    );

    private titleOfDuration = (
        <Title1 text="Trial duration" icon="2.png" />
    );

    private titleOfIntermediate = (
        <div className="panelTitle">
            <Icon type="line-chart" />
            <span>Intermediate result</span>
        </div>
    );

    constructor(props: TrialsDetailProps) {
        super(props);
        this.state = {
            tablePageSize: 20,
            whichGraph: '1',
            searchType: 'id',
            searchFilter: trial => true,
        };
    }

    // search a trial by trial No. & trial id
    searchTrial = (event: React.ChangeEvent<HTMLInputElement>) => {
        const targetValue = event.target.value;
        let filter = (trial: Trial) => true;
        if (!targetValue.trim()) {
            this.setState({ searchFilter: filter });
            return;
        }
        switch (this.state.searchType) {
            case 'id':
                filter = trial => trial.info.id.toUpperCase().includes(targetValue.toUpperCase());
                break;
            case 'Trial No.':
                filter = trial => trial.info.sequenceId.toString() === targetValue;
                break;
            case 'status':
                filter = trial => trial.info.status.toUpperCase().includes(targetValue.toUpperCase());
                break;
            case 'parameters':
                // TODO: support filters like `x: 2` (instead of `"x": 2`)
                filter = trial => JSON.stringify(trial.info.hyperParameters, null, 4).includes(targetValue);
                break;
            default:
                alert(`Unexpected search filter ${this.state.searchType}`);
        }
        this.setState({ searchFilter: filter });
    }

    handleTablePageSizeSelect = (value: string) => {
        this.setState({ tablePageSize: value === 'all' ? -1 : parseInt(value, 10) });
    }

    handleWhichTabs = (activeKey: string) => {
        this.setState({ whichGraph: activeKey });
    }

    updateSearchFilterType = (value: string) => {
        // clear input value and re-render table
        if (this.searchInput !== null) {
            this.searchInput.value = '';
        }
        this.setState({ searchType: value });
    }

    render() {
        const { tablePageSize, whichGraph } = this.state;
        const { columnList, changeColumn } = this.props;
        const source = TRIALS.filter(this.state.searchFilter);
        const trialIds = TRIALS.filter(this.state.searchFilter).map(trial => trial.id);
        return (
            <div>
                <div className="trial" id="tabsty">
                    <Tabs type="card" onChange={this.handleWhichTabs}>
                        <TabPane tab={this.titleOfacc} key="1">
                            <Row className="graph">
                                <DefaultPoint
                                    trialIds={trialIds}
                                    visible={whichGraph === '1'}
                                    trialsUpdateBroadcast={this.props.trialsUpdateBroadcast}
                                />
                            </Row>
                        </TabPane>
                        <TabPane tab={this.titleOfhyper} key="2">
                            <Row className="graph">
                                <Para
                                    dataSource={source}
                                    expSearchSpace={JSON.stringify(EXPERIMENT.searchSpace)}
                                    whichGraph={whichGraph}
                                />
                            </Row>
                        </TabPane>
                        <TabPane tab={this.titleOfDuration} key="3">
                            <Duration source={source} whichGraph={whichGraph} />
                        </TabPane>
                        <TabPane tab={this.titleOfIntermediate} key="4">
                            <Intermediate source={source} whichGraph={whichGraph} />
                        </TabPane>
                    </Tabs>
                </div>
                {/* trial table list */}
                <Title1 text="Trial jobs" icon="6.png" />
                <Row className="allList">
                    <Col span={10}>
                        <span>Show</span>
                        <Select
                            className="entry"
                            onSelect={this.handleTablePageSizeSelect}
                            defaultValue="20"
                        >
                            <Option value="20">20</Option>
                            <Option value="50">50</Option>
                            <Option value="100">100</Option>
                            <Option value="all">All</Option>
                        </Select>
                        <span>entries</span>
                    </Col>
                    <Col span={14} className="right">
                        <Button
                            className="common"
                            onClick={() => { if (this.tableList) { this.tableList.addColumn(); }}}
                        >
                            Add column
                        </Button>
                        <Button
                            className="mediateBtn common"
                            // use child-component tableList's function, the function is in child-component.
                            onClick={() => { if (this.tableList) { this.tableList.compareBtn(); }}}
                        >
                            Compare
                        </Button>
                        <Select defaultValue="id" className="filter" onSelect={this.updateSearchFilterType}>
                            <Option value="id">Id</Option>
                            <Option value="Trial No.">Trial No.</Option>
                            <Option value="status">Status</Option>
                            <Option value="parameters">Parameters</Option>
                        </Select>
                        <input
                            type="text"
                            className="search-input"
                            placeholder={`Search by ${this.state.searchType}`}
                            onChange={this.searchTrial}
                            style={{ width: 230 }}
                            ref={text => (this.searchInput) = text}
                        />
                    </Col>
                </Row>
                <TableList
                    pageSize={tablePageSize}
                    tableSource={source.map(trial => trial.tableRecord)}
                    columnList={columnList}
                    changeColumn={changeColumn}
                    trialsUpdateBroadcast={this.props.trialsUpdateBroadcast}
                    ref={(tabList) => this.tableList = tabList}
                />
            </div>
        );
    }
}

export default TrialsDetail;
