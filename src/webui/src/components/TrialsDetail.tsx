import * as React from 'react';
import axios from 'axios';
import { MANAGER_IP } from '../static/const';
import { Row, Col, Tabs, Input, Select, Button, Icon } from 'antd';
const Option = Select.Option;
import { TableObj, Parameters } from '../static/interface';
import { getFinal } from '../static/function';
import DefaultPoint from './trial-detail/DefaultMetricPoint';
import Duration from './trial-detail/Duration';
import Title1 from './overview/Title1';
import Para from './trial-detail/Para';
import Intermediate from './trial-detail/Intermeidate';
import TableList from './trial-detail/TableList';
const TabPane = Tabs.TabPane;
import '../static/style/trialsDetail.scss';

interface TrialDetailState {
    accSource: object;
    accNodata: string;
    tableListSource: Array<TableObj>;
    searchResultSource: Array<TableObj>;
    isHasSearch: boolean;
    experimentStatus: string;
    experimentPlatform: string;
    experimentLogCollection: boolean;
    entriesTable: number;
    searchSpace: string;
    defaultMetric?: Array<number>;
}

class TrialsDetail extends React.Component<{}, TrialDetailState> {

    public _isMounted = false;
    public interAccuracy = 0;
    public interTableList = 1;
    public interAllTableList = 2;

    public tableList: TableList | null;

    private titleOfacc = (
        <Title1 text="Default Metric" icon="3.png" />
    );

    private titleOfhyper = (
        <Title1 text="Hyper Parameter" icon="1.png" />
    );

    private titleOfDuration = (
        <Title1 text="Trial Duration" icon="2.png" />
    );

    private titleOfIntermediate = (
        <div className="panelTitle">
            <Icon type="line-chart" />
            <span>Intermediate Result</span>
        </div>
    );

    constructor(props: {}) {
        super(props);

        this.state = {
            accSource: {},
            accNodata: '',
            tableListSource: [],
            searchResultSource: [],
            experimentStatus: '',
            experimentPlatform: '',
            experimentLogCollection: false,
            entriesTable: 20,
            isHasSearch: false,
            searchSpace: '',
            defaultMetric: [0, 1]
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
                if (res.status === 200) {
                    const trialJobs = res.data;
                    const metricSource = res1.data;
                    const trialTable: Array<TableObj> = [];
                    Object.keys(trialJobs).map(item => {
                        // only succeeded trials have finalMetricData
                        let desc: Parameters = {
                            parameters: {},
                            intermediate: []
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
                        if (trialJobs[item].hyperParameters !== undefined) {
                            const getPara = JSON.parse(trialJobs[item].hyperParameters[0]).parameters;
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
                    // get acc max and min for hyper-parameters graph color bar max and min
                    const sortSource = JSON.parse(JSON.stringify(trialTable));
                    sortSource.sort((a: TableObj, b: TableObj) => {
                        if (a.acc !== undefined && b.acc !== undefined) {
                            return JSON.parse(a.acc.default) - JSON.parse(b.acc.default);
                        } else {
                            return NaN;
                        }
                    });

                    if (this._isMounted && sortSource !== undefined) {

                        const hyperMin = sortSource[0].acc !== undefined
                            ?
                            sortSource[0].acc.default
                            :
                            '0';
                        const max = sortSource[sortSource.length - 1].acc;
                        let maxResult = '1';
                        if (max !== undefined) {
                            maxResult = max.default;
                        }
                        this.setState(() => ({
                            defaultMetric: [JSON.parse(hyperMin), JSON.parse(maxResult)]
                        }));
                    }
                    // update search data result
                    const { searchResultSource } = this.state;
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
                        this.setState(() => ({
                            tableListSource: trialTable
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
            const { tableListSource } = this.state;
            const searchResultList: Array<TableObj> = [];
            Object.keys(tableListSource).map(key => {
                const item = tableListSource[key];
                if (item.sequenceId.toString() === targetValue
                    || item.id.includes(targetValue)
                    || item.status.includes(targetValue)
                ) {
                    searchResultList.push(item);
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
        axios(`${MANAGER_IP}/check-status`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200 && this._isMounted) {
                    switch (res.data.status) {
                        case 'DONE':
                        case 'ERROR':
                        case 'STOPPED':
                            window.clearInterval(this.interTableList);
                            break;
                        default:
                    }
                }
            });
    }

    handleEntriesSelect = (value: string) => {
        switch (value) {
            case '20':
                this.setState(() => ({ entriesTable: 20 }));
                break;
            case '50':
                this.setState(() => ({ entriesTable: 50 }));
                break;
            case '100':
                this.setState(() => ({ entriesTable: 100 }));
                break;
            case 'all':
                const { tableListSource } = this.state;
                this.setState(() => ({ entriesTable: tableListSource.length }));
                break;
            default:
        }
    }

    test = () => {
        alert('TableList component was not properly initialized.');
    }

    // get and set logCollection val
    checkExperimentPlatform = () => {
        axios(`${MANAGER_IP}/experiment`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    const trainingPlatform = res.data.params.trainingServicePlatform !== undefined
                        ?
                        res.data.params.trainingServicePlatform
                        :
                        '';
                    // default logCollection is true
                    const logCollection = res.data.params.logCollection;
                    let expLogCollection: boolean = false;
                    if (logCollection !== undefined && logCollection !== 'none') {
                        expLogCollection = true;
                    }
                    if (this._isMounted) {
                        this.setState({
                            experimentPlatform: trainingPlatform,
                            searchSpace: res.data.params.searchSpace,
                            experimentLogCollection: expLogCollection
                        });
                    }
                }
            });
    }

    componentDidMount() {

        this._isMounted = true;
        this.getDetailSource();
        this.interTableList = window.setInterval(this.getDetailSource, 10000);
        this.checkExperimentPlatform();
    }

    componentWillUnmount() {
        this._isMounted = false;
        window.clearInterval(this.interTableList);
    }

    render() {

        const {
            tableListSource, searchResultSource, isHasSearch,
            entriesTable, experimentPlatform, searchSpace,
            defaultMetric, experimentLogCollection
        } = this.state;
        const source = isHasSearch ? searchResultSource : tableListSource;
        return (
            <div>
                <div className="trial" id="tabsty">
                    <Tabs type="card">
                        <TabPane tab={this.titleOfacc} key="1">
                            <Row className="graph">
                                <DefaultPoint
                                    height={432}
                                    showSource={source}
                                />
                            </Row>
                        </TabPane>
                        <TabPane tab={this.titleOfhyper} key="2">
                            <Row className="graph">
                                <Para
                                    dataSource={source}
                                    expSearchSpace={searchSpace}
                                    defaultMetric={defaultMetric}
                                />
                            </Row>
                        </TabPane>
                        <TabPane tab={this.titleOfDuration} key="3">
                            <Duration source={source} />
                        </TabPane>
                        <TabPane tab={this.titleOfIntermediate} key="4">
                            <Intermediate source={source} />
                        </TabPane>
                    </Tabs>
                </div>
                {/* trial table list */}
                <Title1 text="Trial Jobs" icon="6.png" />
                <Row className="allList">
                    <Col span={12}>
                        <span>show</span>
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
                    <Col span={12} className="right">
                        <Row>
                            <Col span={12}>
                                <Button
                                    type="primary"
                                    className="tableButton editStyle"
                                    onClick={this.tableList ? this.tableList.addColumn : this.test}
                                >
                                    AddColumn
                                </Button>
                            </Col>
                            <Col span={12}>
                                <Input
                                    type="text"
                                    placeholder="search by Id, Trial No. or Status"
                                    onChange={this.searchTrial}
                                    style={{ width: 230, marginLeft: 6 }}
                                />
                            </Col>
                        </Row>
                    </Col>
                </Row>
                <TableList
                    entries={entriesTable}
                    tableSource={source}
                    platform={experimentPlatform}
                    updateList={this.getDetailSource}
                    logCollection={experimentLogCollection}
                    ref={(tabList) => this.tableList = tabList}
                />
            </div>
        );
    }
}

export default TrialsDetail;