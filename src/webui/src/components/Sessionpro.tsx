import * as React from 'react';
import axios from 'axios';
import { Table, Select, Row, Col, Icon } from 'antd';
import { MANAGER_IP, overviewItem, roundNum } from '../const';
import ReactEcharts from 'echarts-for-react';
const Option = Select.Option;
import JSONTree from 'react-json-tree';
require('echarts/lib/chart/line');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');
require('../style/sessionpro.css');

interface TableObj {
    key: number;
    id: string;
    duration: number;
    start: string;
    end: string;
    status: string;
    acc?: number;
    description: object;
}

interface Parameters {
    parameters: object;
    logPath?: string;
}

interface Experiment {
    id: string;
    author: string;
    experName: string;
    runConcurren: number;
    maxDuration: number;
    execDuration: number;
    MaxTrialNum: number;
    startTime: string;
    endTime: string;
}

interface SessionState {
    tableData: Array<TableObj>;
    searchSpace: object;
    trialProfile: Experiment;
    tunerAssessor: object;
    selNum: number;
    selStatus: string;
    trialRun: Array<number>;
    option: object;
    noData: string;
}

class Sessionpro extends React.Component<{}, SessionState> {

    public _isMounted = false;
    public intervalID = 0;
    public intervalProfile = 1;

    constructor(props: {}) {
        super(props);
        this.state = {
            searchSpace: {},
            trialProfile: {
                id: '',
                author: '',
                experName: '',
                runConcurren: 0,
                maxDuration: 0,
                execDuration: 0,
                MaxTrialNum: 0,
                startTime: '',
                endTime: ''
            },
            tunerAssessor: {},
            tableData: [{
                key: 0,
                id: '',
                duration: 0,
                start: '',
                end: '',
                status: '',
                acc: 0,
                description: {}
            }],
            selNum: overviewItem,
            selStatus: 'Complete',
            trialRun: [],
            option: {},
            noData: ''
        };
    }

    sortNumber = (a: number, b: number) => {

        return a - b;
    }

    // draw cdf data
    getOption = (data: Array<number>) => {
        let len = data.length;
        // let min = Math.floor(Math.min.apply(null, data));
        let min = Math.floor(data[0]);
        let max = Math.ceil(data[len - 1]);
        let gap = (max - min) / 10;
        let a = 0;
        let b = 0;
        let c = 0;
        let d = 0;
        let e = 0;
        let f = 0;
        let g = 0;
        let h = 0;
        let i = 0;
        let j = 0;

        let xAxis: number[] = [];
        for (let m = 0; m < 10; m++) {
            xAxis.push(min + gap * m);
        }

        data.map(item => {
            switch (Math.floor((item - min) / gap)) {

                case 0: a++; b++; c++; d++; e++; f++; g++; h++; i++; j++; break;
                case 1: b++; c++; d++; e++; f++; g++; h++; i++; j++; break;
                case 2: c++; d++; e++; f++; g++; h++; i++; j++; break;
                case 3: d++; e++; f++; g++; h++; i++; j++; break;
                case 4: e++; f++; g++; h++; i++; j++; break;
                case 5: f++; g++; h++; i++; j++; break;
                case 6: g++; h++; i++; j++; break;
                case 7: h++; i++; j++; break;
                case 8: i++; j++; break;
                case 9: j++; break;
                default: j++; break;
            }
        });
        let prob = [a / len, b / len, c / len, d / len, e / len, f / len, g / len, h / len, i / len, j / len];
        return {
            tooltip: {
                trigger: 'item'
            },
            title: {
                left: 'center',
                text: 'Succeeded Trials CDF',
                top: 16
            },
            grid: {
                left: '5%'
            },
            xAxis: {
                name: 'trial running time/s',
                type: 'category',
                data: xAxis
            },
            yAxis: {
                name: 'percent',
                type: 'value',
                min: 0,
                max: 1
            },
            series: [
                {
                    type: 'line',
                    smooth: true,
                    itemStyle: {
                        normal: {
                            color: 'skyblue'
                        }
                    },
                    data: prob
                }
            ]
        };
    }

    // show session
    showSessionPro = () => {
        axios(`${MANAGER_IP}/experiment`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    let sessionData = res.data;
                    let tunerAsstemp = [];
                    let trialPro = [];
                    trialPro.push({
                        id: sessionData.id,
                        author: sessionData.params.authorName,
                        experName: sessionData.params.experimentName,
                        runConcurren: sessionData.params.trialConcurrency,
                        maxDuration: sessionData.params.maxExecDuration,
                        execDuration: sessionData.execDuration,
                        MaxTrialNum: sessionData.params.maxTrialNum,
                        startTime: sessionData.startTime,
                        endTime: sessionData.endTime === undefined ? 'not over' : sessionData.endTime
                    });
                    tunerAsstemp.push({
                        tuner: sessionData.params.tuner,
                        assessor: sessionData.params.assessor
                    });
                    if (this._isMounted) {
                        this.setState({
                            trialProfile: trialPro[0],
                            searchSpace: JSON.parse(sessionData.params.searchSpace),
                            tunerAssessor: tunerAsstemp[0]
                        });
                    }
                }
            });
    }

    showTrials = () => {
        axios(`${MANAGER_IP}/trial-jobs`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    // deal with complete trial data to draw CDF graph
                    let trialRunData: Array<number> = [];
                    const { selNum } = this.state;
                    const tableData = res.data;
                    const topTableData: Array<TableObj> = [];
                    Object.keys(tableData).map(item => {
                        if (tableData[item].status === 'SUCCEEDED') {
                            const desJobDetail: Parameters = {
                                parameters: {}
                            };
                            const startTime = Date.parse(tableData[item].startTime);
                            const duration = (Date.parse(tableData[item].endTime) - startTime) / 1000;
                            let acc;
                            if (tableData[item].finalMetricData) {
                                const accFloat = parseFloat(tableData[item].finalMetricData.data);
                                acc = roundNum(accFloat, 5);
                            } else {
                                acc = 0;
                            }
                            desJobDetail.parameters = JSON.parse(tableData[item].hyperParameters).parameters;
                            if (tableData[item].logPath !== undefined) {
                                desJobDetail.logPath = tableData[item].logPath;
                            }
                            topTableData.push({
                                key: topTableData.length,
                                id: tableData[item].id,
                                duration: duration,
                                start: tableData[item].startTime,
                                end: tableData[item].endTime,
                                status: tableData[item].status,
                                acc: acc,
                                description: desJobDetail
                            });
                            trialRunData.push(duration);
                        }
                    });
                    topTableData.sort((a: TableObj, b: TableObj) => {
                        if (a.acc && b.acc) {
                            return b.acc - a.acc;
                        } else {
                            return NaN;
                        }
                    });
                    topTableData.length = Math.min(selNum, topTableData.length);
                    if (this._isMounted) {
                        this.setState({
                            tableData: topTableData,
                            trialRun: trialRunData.sort(this.sortNumber)
                        });
                    }
                    // draw CDF
                    const { trialRun } = this.state;
                    if (this._isMounted) {
                        this.setState({
                            option: this.getOption(trialRun)
                        });
                    }
                    // CDF graph 'No data' judge
                    if (trialRun.length === 0) {
                        if (this._isMounted) {
                            this.setState({
                                noData: 'No data'
                            });
                        }
                    } else {
                        if (this._isMounted) {
                            this.setState({
                                noData: ''
                            });
                        }
                    }
                }
            });
    }

    handleChange = (value: string) => {
        let num = parseFloat(value);
        window.clearInterval(this.intervalID);
        if (this._isMounted) {
            this.setState({ selNum: num }, () => {
                this.showTrials();
                this.intervalID = window.setInterval(this.showTrials, 60000);
            });
        }
    }

    componentDidMount() {
        this.showSessionPro();
        this.showTrials();
        this._isMounted = true;
        this.intervalID = window.setInterval(this.showTrials, 10000);
        this.intervalProfile = window.setInterval(this.showSessionPro, 60000);
    }

    componentWillUnmount() {
        this._isMounted = false;
        window.clearInterval(this.intervalID);
        window.clearInterval(this.intervalProfile);
    }

    render() {
        // show better job details
        let bgColor = '';
        const columns = [{
            title: 'Id',
            dataIndex: 'id',
            key: 'id',
            width: 150,
            className: 'tableHead',
        }, {
            title: 'Duration/s',
            dataIndex: 'duration',
            key: 'duration',
            width: '9%'
        }, {
            title: 'Start',
            dataIndex: 'start',
            key: 'start',
            width: 150
        }, {
            title: 'End',
            dataIndex: 'end',
            key: 'end',
            width: 150
        }, {
            title: 'Status',
            dataIndex: 'status',
            key: 'status',
            width: 150,
            className: 'tableStatus',
            render: (text: string, record: TableObj) => {
                bgColor = record.status;
                return (
                    <span className={`${bgColor} commonStyle`}>{record.status}</span>
                );
            }
        }, {
            title: 'Loss/Accuracy',
            dataIndex: 'acc',
            key: 'acc',
            width: 150
        }];

        const openRow = (record: TableObj) => {
            return (
                <pre id="description" className="jsontree">
                    <JSONTree
                        hideRoot={true}
                        shouldExpandNode={() => true}  // default expandNode
                        getItemString={() => (<span />)}  // remove the {} items
                        data={record.description}
                    />
                </pre>
            );
        };

        const {
            trialProfile, searchSpace, tunerAssessor, tableData, option, noData
        } = this.state;
        let running;
        if (trialProfile.endTime === 'not over') {
            running = trialProfile.maxDuration - trialProfile.execDuration;
        } else {
            running = 0;
        }
        return (
            <div className="session" id="session">
                <div className="head">
                    <div className="headCon">
                        <div className="author">
                            <div className="message">
                                <div className="proKey">
                                    <span>Author</span>
                                    <span className="messcont">{trialProfile.author}</span>
                                </div>
                                <span>Experiment&nbsp;Name</span>
                                <p className="messcont">{trialProfile.experName}</p>
                            </div>
                            <div className="logo">
                                <Icon className="bone" type="user" />
                            </div>
                        </div>
                        <div className="type">
                            <div className="message">
                                <div className="proKey">
                                    <span>id</span>
                                    <span className="messcont">{trialProfile.id}</span>
                                </div>
                                <p>
                                    <span>Duration</span>
                                    <span className="messcont">{trialProfile.maxDuration}s</span>
                                </p>
                                <p>
                                    <span>Still&nbsp;running</span>
                                    <span className="messcont">{running}s</span>
                                </p>
                            </div>
                            <div className="logo">
                                <Icon className="tyellow" type="bulb" />
                            </div>
                        </div>
                        <div className="runtime message">
                            <p className="proTime">
                                <span>Start Time</span><br />
                                <span className="messcont">{trialProfile.startTime}</span>
                            </p>
                            <span>End Time</span>
                            <p className="messcont">{trialProfile.endTime}</p>
                            {/* <div className="logo">
                                <Icon className="thrpink" type="clock-circle-o" />
                            </div> */}
                        </div>
                        <div className="cdf">
                            <div className="message">
                                <div className="proKey trialNum">
                                    Concurrency&nbsp;Trial
                                    <span className="messcont">{trialProfile.runConcurren}</span>
                                </div>
                                <p>
                                    Max&nbsp;Trial&nbsp;Number
                                    <span className="messcont">{trialProfile.MaxTrialNum}</span>
                                </p>
                            </div>
                            <div className="logo">
                                <Icon className="fogreen" type="picture" />
                            </div>
                        </div>
                    </div>
                </div>
                <div className="clear" />
                <div className="jsonbox">
                    <div>
                        <h2 className="searchTitle title">Search Space</h2>
                        <pre className="searchSpace jsontree">
                            <JSONTree
                                hideRoot={true}
                                shouldExpandNode={() => true}
                                getItemString={() => (<span />)}
                                data={searchSpace}
                            />
                        </pre>
                    </div>
                    <div>
                        <h2 className="searchTitle title">Trial Profile</h2>
                        <pre className="trialProfile jsontree">
                            <JSONTree
                                hideRoot={true}
                                shouldExpandNode={() => true}
                                getItemString={() => (<span />)}
                                data={tunerAssessor}
                            />
                        </pre>
                    </div>
                </div>
                <div className="clear" />
                <div className="comtable">
                    <div className="selectInline">
                        <Row>
                            <Col span={18}>
                                <h2>The trials that successed</h2>
                            </Col>
                            <Col span={6}>
                                <span className="tabuser1">top</span>
                                <Select
                                    style={{ width: 200 }}
                                    placeholder="5"
                                    optionFilterProp="children"
                                    onSelect={this.handleChange}
                                >
                                    <Option value="20">20</Option>
                                    <Option value="50">50</Option>
                                    <Option value="100">100</Option>
                                </Select>
                            </Col>
                        </Row>
                    </div>
                    <Table
                        columns={columns}
                        expandedRowRender={openRow}
                        dataSource={tableData}
                        className="tables"
                        bordered={true}
                        scroll={{ x: '100%', y: 540 }}
                    />
                </div>
                <div className="cdf">
                    <ReactEcharts
                        option={option}
                        style={{ height: 500, padding: '0px' }}
                    />
                    <div className="addNodata">{noData}</div>
                </div>
            </div>
        );
    }
}
export default Sessionpro;
