import * as React from 'react';
import axios from 'axios';
import { MANAGER_IP } from '../static/const';
import { Row, Col, Tabs, Select, Button, Icon } from 'antd';
const Option = Select.Option;
import { TableObj, Parameters, ExperimentInfo } from '../static/interface';
import { getFinal } from '../static/function';
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
    accSource: object;
    accNodata: string;
    tableListSource: Array<TableObj>;
    searchResultSource: Array<TableObj>;
    isHasSearch: boolean;
    experimentLogCollection: boolean;
    entriesTable: number; // table components val
    entriesInSelect: string;
    searchSpace: string;
    isMultiPhase: boolean;
    whichGraph: string;
    hyperCounts: number; // user click the hyper-parameter counts
    durationCounts: number;
    intermediateCounts: number;
    experimentInfo: ExperimentInfo;
    searchFilter: string;
    searchPlaceHolder: string;
}

interface TrialsDetailProps {
    interval: number;
    whichPageToFresh: string;
    columnList: Array<string>;
    changeColumn: (val: Array<string>) => void;
}

class TrialsDetail extends React.Component<TrialsDetailProps, TrialDetailState> {

    public _isMounted = false;
    public interAccuracy = 0;
    public interTableList = 1;
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
            accSource: {},
            accNodata: '',
            tableListSource: [],
            searchResultSource: [],
            experimentLogCollection: false,
            entriesTable: 20,
            entriesInSelect: '20',
            searchSpace: '',
            whichGraph: '1',
            isHasSearch: false,
            isMultiPhase: false,
            hyperCounts: 0,
            durationCounts: 0,
            intermediateCounts: 0,
            experimentInfo: {
                platform: '',
                optimizeMode: 'maximize'
            },
            searchFilter: 'id',
            searchPlaceHolder: 'Search by id'
        };
    }

    getDetailSource = () => {
        this.isOffIntervals();
        axios
            .all([
                axios.get(`${MANAGER_IP}/trial-jobs`),
                axios.get(`${MANAGER_IP}/metric-data`)
            ])
            .then(axios.spread((res, res1) => {
                if (res.status === 200 && res1.status === 200) {
                    const trialJobs = res.data;
                    const metricSource = res1.data;
                    const trialTable: Array<TableObj> = [];
                    Object.keys(trialJobs).map(item => {
                        let desc: Parameters = {
                            parameters: {},
                            intermediate: [],
                            multiProgress: 1
                        };
                        let duration = 0;
                        const id = trialJobs[item].id !== undefined
                            ? trialJobs[item].id
                            : '';
                        const status = trialJobs[item].status !== undefined
                            ? trialJobs[item].status
                            : '';
                        const begin = trialJobs[item].startTime;
                        const end = trialJobs[item].endTime;
                        if (begin) {
                            if (end) {
                                duration = (end - begin) / 1000;
                            } else {
                                duration = (new Date().getTime() - begin) / 1000;
                            }
                        }
                        const tempHyper = trialJobs[item].hyperParameters;
                        if (tempHyper !== undefined) {
                            const getPara = JSON.parse(tempHyper[tempHyper.length - 1]).parameters;
                            desc.multiProgress = tempHyper.length;
                            if (typeof getPara === 'string') {
                                desc.parameters = JSON.parse(getPara);
                            } else {
                                desc.parameters = getPara;
                            }
                        } else {
                            desc.parameters = { error: 'This trial\'s parameters are not available.' };
                        }
                        if (trialJobs[item].logPath !== undefined) {
                            desc.logPath = trialJobs[item].logPath;
                        }

                        const acc = getFinal(trialJobs[item].finalMetricData);
                        // deal with intermediate result list
                        const mediate: Array<number> = [];
                        Object.keys(metricSource).map(key => {
                            const items = metricSource[key];
                            if (items.trialJobId === id) {
                                // succeed trial, last intermediate result is final result
                                // final result format may be object
                                if (typeof JSON.parse(items.data) === 'object') {
                                    mediate.push(JSON.parse(items.data).default);
                                } else {
                                    mediate.push(JSON.parse(items.data));
                                }
                            }
                        });
                        desc.intermediate = mediate;
                        trialTable.push({
                            key: trialTable.length,
                            sequenceId: trialJobs[item].sequenceId,
                            id: id,
                            status: status,
                            duration: duration,
                            acc: acc,
                            description: desc
                        });
                    });
                    // update search data result
                    const { searchResultSource, entriesInSelect } = this.state;
                    if (searchResultSource.length !== 0) {
                        const temp: Array<number> = [];
                        Object.keys(searchResultSource).map(index => {
                            temp.push(searchResultSource[index].id);
                        });
                        const searchResultList: Array<TableObj> = [];
                        for (let i = 0; i < temp.length; i++) {
                            Object.keys(trialTable).map(key => {
                                const item = trialTable[key];
                                if (item.id === temp[i]) {
                                    searchResultList.push(item);
                                }
                            });
                        }

                        if (this._isMounted) {
                            this.setState(() => ({
                                searchResultSource: searchResultList
                            }));
                        }
                    }
                    if (this._isMounted) {
                        this.setState(() => ({ tableListSource: trialTable }));
                    }
                    if (entriesInSelect === 'all' && this._isMounted) {
                        this.setState(() => ({
                            entriesTable: trialTable.length
                        }));
                    }
                }
            }));
    }

    // search a trial by trial No. & trial id
    searchTrial = (event: React.ChangeEvent<HTMLInputElement>) => {
        const targetValue = event.target.value;
        if (targetValue === '' || targetValue === ' ') {
            const { tableListSource } = this.state;
            if (this._isMounted) {
                this.setState(() => ({
                    isHasSearch: false,
                    tableListSource: tableListSource,
                }));
            }
        } else {
            const { tableListSource, searchFilter } = this.state;
            const searchResultList: Array<TableObj> = [];
            Object.keys(tableListSource).map(key => {
                const item = tableListSource[key];
                switch (searchFilter) {
                    case 'id':
                        if (item.id.toUpperCase().includes(targetValue.toUpperCase())) {
                            searchResultList.push(item);
                        }
                        break;
                    case 'Trial No.':
                        if (item.sequenceId.toString() === targetValue) {
                            searchResultList.push(item);
                        }
                        break;
                    case 'status':
                        if (item.status.toUpperCase().includes(targetValue.toUpperCase())) {
                            searchResultList.push(item);
                        }
                        break;
                    case 'parameters':
                        const strParameters = JSON.stringify(item.description.parameters, null, 4);
                        if (strParameters.includes(targetValue)) {
                            searchResultList.push(item);
                        }
                        break;
                    default:
                }
            });
            if (this._isMounted) {
                this.setState(() => ({
                    searchResultSource: searchResultList,
                    isHasSearch: true
                }));
            }
        }
    }

    // close timer
    isOffIntervals = () => {
        const { interval } = this.props;
        if (interval === 0) {
            window.clearInterval(this.interTableList);
            return;
        } else {
            axios(`${MANAGER_IP}/check-status`, {
                method: 'GET'
            })
                .then(res => {
                    if (res.status === 200 && this._isMounted) {
                        const expStatus = res.data.status;
                        if (expStatus === 'DONE' || expStatus === 'ERROR' || expStatus === 'STOPPED') {
                            window.clearInterval(this.interTableList);
                            return;
                        }
                    }
                });
        }
    }

    handleEntriesSelect = (value: string) => {
        // user select isn't 'all'
        if (value !== 'all') {
            if (this._isMounted) {
                this.setState(() => ({ entriesTable: parseInt(value, 10) }));
            }
        } else {
            const { tableListSource } = this.state;
            if (this._isMounted) {
                this.setState(() => ({
                    entriesInSelect: 'all',
                    entriesTable: tableListSource.length
                }));
            }
        }
    }

    handleWhichTabs = (activeKey: string) => {
        // const which = JSON.parse(activeKey);
        if (this._isMounted) {
            this.setState(() => ({ whichGraph: activeKey }));
        }
    }

    test = () => {
        alert('TableList component was not properly initialized.');
    }

    getSearchFilter = (value: string) => {
        // clear input value and re-render table
        if (this.searchInput !== null) {
            this.searchInput.value = '';
            if (this._isMounted === true) {
                this.setState(() => ({ isHasSearch: false }));
            }
        }
        if (this._isMounted === true) {
            this.setState(() => ({ searchFilter: value, searchPlaceHolder: `Search by ${value}` }));
        }
    }

    // get and set logCollection val
    checkExperimentPlatform = () => {
        axios(`${MANAGER_IP}/experiment`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    const trainingPlatform: string = res.data.params.trainingServicePlatform !== undefined
                        ?
                        res.data.params.trainingServicePlatform
                        :
                        '';
                    // default logCollection is true
                    const logCollection = res.data.params.logCollection;
                    let expLogCollection: boolean = false;
                    const isMultiy: boolean = res.data.params.multiPhase !== undefined
                        ? res.data.params.multiPhase : false;
                    const tuner = res.data.params.tuner;
                    // I'll set optimize is maximize if user not set optimize
                    let optimize: string = 'maximize';
                    if (tuner !== undefined) {
                        if (tuner.classArgs !== undefined) {
                            if (tuner.classArgs.optimize_mode !== undefined) {
                                if (tuner.classArgs.optimize_mode === 'minimize') {
                                    optimize = 'minimize';
                                }
                            }
                        }
                    }
                    if (logCollection !== undefined && logCollection !== 'none') {
                        expLogCollection = true;
                    }
                    if (this._isMounted) {
                        this.setState({
                            experimentInfo: { platform: trainingPlatform, optimizeMode: optimize },
                            searchSpace: res.data.params.searchSpace,
                            experimentLogCollection: expLogCollection,
                            isMultiPhase: isMultiy
                        });
                    }
                }
            });
    }

    componentWillReceiveProps(nextProps: TrialsDetailProps) {
        const { interval, whichPageToFresh } = nextProps;
        window.clearInterval(this.interTableList);
        if (interval !== 0) {
            this.interTableList = window.setInterval(this.getDetailSource, interval * 1000);
        }
        if (whichPageToFresh.includes('/detail')) {
            this.getDetailSource();
        }
    }

    componentDidMount() {

        this._isMounted = true;
        const { interval } = this.props;
        this.getDetailSource();
        this.interTableList = window.setInterval(this.getDetailSource, interval * 1000);
        this.checkExperimentPlatform();
    }

    componentWillUnmount() {
        this._isMounted = false;
        window.clearInterval(this.interTableList);
    }

    render() {

        const {
            tableListSource, searchResultSource, isHasSearch, isMultiPhase,
            entriesTable, experimentInfo, searchSpace, experimentLogCollection,
            whichGraph, searchPlaceHolder
        } = this.state;
        const source = isHasSearch ? searchResultSource : tableListSource;
        const { columnList, changeColumn } = this.props;
        return (
            <div>
                <div className="trial" id="tabsty">
                    <Tabs type="card" onChange={this.handleWhichTabs}>
                        <TabPane tab={this.titleOfacc} key="1">
                            <Row className="graph">
                                <DefaultPoint
                                    height={402}
                                    showSource={source}
                                    whichGraph={whichGraph}
                                    optimize={experimentInfo.optimizeMode}
                                />
                            </Row>
                        </TabPane>
                        <TabPane tab={this.titleOfhyper} key="2">
                            <Row className="graph">
                                <Para
                                    dataSource={source}
                                    expSearchSpace={searchSpace}
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
                            onSelect={this.handleEntriesSelect}
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
                            onClick={this.tableList ? this.tableList.addColumn : this.test}
                        >
                            Add column
                        </Button>
                        <Button
                            className="mediateBtn common"
                            // use child-component tableList's function, the function is in child-component.
                            onClick={this.tableList ? this.tableList.compareBtn : this.test}
                        >
                            Compare
                        </Button>
                        <Select defaultValue="id" className="filter" onSelect={this.getSearchFilter}>
                            <Option value="id">Id</Option>
                            <Option value="Trial No.">Trial No.</Option>
                            <Option value="status">Status</Option>
                            <Option value="parameters">Parameters</Option>
                        </Select>
                        <input
                            type="text"
                            className="search-input"
                            placeholder={searchPlaceHolder}
                            onChange={this.searchTrial}
                            style={{ width: 230 }}
                            ref={text => (this.searchInput) = text}
                        />
                    </Col>
                </Row>
                <TableList
                    entries={entriesTable}
                    tableSource={source}
                    isMultiPhase={isMultiPhase}
                    platform={experimentInfo.platform}
                    updateList={this.getDetailSource}
                    logCollection={experimentLogCollection}
                    columnList={columnList}
                    changeColumn={changeColumn}
                    ref={(tabList) => this.tableList = tabList}
                />
            </div>
        );
    }
}

export default TrialsDetail;