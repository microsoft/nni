import * as React from 'react';
import axios from 'axios';
import { Table, Select, Row, Col, Icon, Button } from 'antd';
import { MANAGER_IP, overviewItem } from '../const';
const Option = Select.Option;
import JSONTree from 'react-json-tree';
require('../style/sessionpro.css');
require('../style/logPath.css');

interface TableObj {
    key: number;
    id: string;
    duration: number;
    start: string;
    end: string;
    status: string;
    acc?: number;
    description: Parameters;
}

interface Parameters {
    parameters: object;
    logPath?: string;
    isLink?: boolean;
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
    status: string;
    trialProfile: Experiment;
    tunerAssessor: object;
    selNum: number;
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
            status: '',
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
                description: {
                    parameters: {}
                }
            }],
            selNum: overviewItem,
            option: {},
            noData: ''
        };
    }

    convertTime = (num: number) => {
        if (num % 3600 === 0) {
            return num / 3600 + 'h';
        } else {
            const hour = Math.floor(num / 3600);
            const min = Math.floor(num / 60 % 60);
            return hour > 0 ? `${hour} h ${min} min` : `${min} min`;
        }
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
                    const startExper = new Date(sessionData.startTime).toLocaleString('en-US');
                    let experEndStr: string;
                    if (sessionData.endTime !== undefined) {
                        experEndStr = new Date(sessionData.endTime).toLocaleString('en-US');
                    } else {
                        experEndStr = 'not over';
                    }
                    trialPro.push({
                        id: sessionData.id,
                        author: sessionData.params.authorName,
                        experName: sessionData.params.experimentName,
                        runConcurren: sessionData.params.trialConcurrency,
                        maxDuration: sessionData.params.maxExecDuration,
                        execDuration: sessionData.execDuration,
                        MaxTrialNum: sessionData.params.maxTrialNum,
                        startTime: startExper,
                        endTime: experEndStr
                    });
                    tunerAsstemp.push({
                        tuner: sessionData.params.tuner,
                        assessor: sessionData.params.assessor
                    });
                    // search space format loguniform max and min
                    const searchSpace = JSON.parse(sessionData.params.searchSpace);
                    Object.keys(searchSpace).map(item => {
                        const key = searchSpace[item]._type;
                        if (key === 'loguniform' || key === 'qloguniform') {
                            let value = searchSpace[item]._value;
                            const a = Math.pow(10, value[0]);
                            const b = Math.pow(10, value[1]);
                            if (a < b) {
                                value = [a, b];
                            } else {
                                value = [b, a];
                            }
                            searchSpace[item]._value = value;
                        }
                    });
                    if (this._isMounted) {
                        this.setState({
                            trialProfile: trialPro[0],
                            searchSpace: searchSpace,
                            tunerAssessor: tunerAsstemp[0]
                        });
                    }
                }
            });

        axios(`${MANAGER_IP}/check-status`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200 && this._isMounted) {
                    this.setState({
                        status: res.data.status
                    });
                }
            });
    }

    showTrials = () => {
        axios(`${MANAGER_IP}/trial-jobs`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    const { selNum } = this.state;
                    const tableData = res.data;
                    const topTableData: Array<TableObj> = [];
                    Object.keys(tableData).map(item => {
                        if (tableData[item].status === 'SUCCEEDED') {
                            const desJobDetail: Parameters = {
                                parameters: {}
                            };
                            const startTime = new Date(tableData[item].startTime).toLocaleString('en-US');
                            const endTime = new Date(tableData[item].endTime).toLocaleString('en-US');
                            const duration = (tableData[item].endTime - tableData[item].startTime) / 1000;
                            let acc;
                            if (tableData[item].finalMetricData) {
                                acc = parseFloat(tableData[item].finalMetricData.data);
                            }
                            desJobDetail.parameters = JSON.parse(tableData[item].hyperParameters).parameters;
                            if (tableData[item].logPath !== undefined) {
                                desJobDetail.logPath = tableData[item].logPath;
                                const isSessionLink = /^http/gi.test(tableData[item].logPath);
                                if (isSessionLink) {
                                    desJobDetail.isLink = true;
                                }
                            }
                            topTableData.push({
                                key: topTableData.length,
                                id: tableData[item].id,
                                duration: duration,
                                start: startTime,
                                end: endTime,
                                status: tableData[item].status,
                                acc: acc,
                                description: desJobDetail
                            });
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
                            tableData: topTableData
                        });
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
                this.intervalID = window.setInterval(this.showTrials, 10000);
            });
        }
    }

    downExperimentContent = () => {
        axios
            .all([
                axios.get(`${MANAGER_IP}/experiment`),
                axios.get(`${MANAGER_IP}/trial-jobs`),
                axios.get(`${MANAGER_IP}/metric-data`)
            ])
            .then(axios.spread((res, res1, res2) => {
                if (res.status === 200 && res1.status === 200 && res2.status === 200) {
                    if (res.data.params.searchSpace) {
                        res.data.params.searchSpace = JSON.parse(res.data.params.searchSpace);
                    }
                    const interResultList = res2.data;
                    const contentOfExperiment = JSON.stringify(res.data, null, 2);
                    let trialMessagesArr = res1.data;
                    Object.keys(trialMessagesArr).map(item => {
                        // transform hyperparameters as object to show elegantly
                        trialMessagesArr[item].hyperParameters = JSON.parse(trialMessagesArr[item].hyperParameters);
                        const trialId = trialMessagesArr[item].id;
                        // add intermediate result message
                        trialMessagesArr[item].intermediate = [];
                        Object.keys(interResultList).map(key => {
                            const interId = interResultList[key].trialJobId;
                            if (trialId === interId) {
                                trialMessagesArr[item].intermediate.push(interResultList[key]);
                            }
                        });
                    });
                    const trialMessages = JSON.stringify(trialMessagesArr, null, 2);
                    const aTag = document.createElement('a');
                    const file = new Blob([contentOfExperiment, trialMessages], { type: 'application/json' });
                    aTag.download = 'experiment.txt';
                    aTag.href = URL.createObjectURL(file);
                    aTag.click();
                    URL.revokeObjectURL(aTag.href);
                    if (navigator.userAgent.indexOf('Firefox') > -1) {
                        const downTag = document.createElement('a');
                        downTag.addEventListener('click', function () {
                            downTag.download = 'experiment.txt';
                            downTag.href = URL.createObjectURL(file);
                        });
                        let eventMouse = document.createEvent('MouseEvents');
                        eventMouse.initEvent('click', false, false);
                        downTag.dispatchEvent(eventMouse);
                    }
                }
            }));
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
            const openRowDataSource = {
                parameters: record.description.parameters
            };
            let isLogLink: boolean = false;
            const logPathRow = record.description.logPath;
            if (record.description.isLink !== undefined) {
                isLogLink = true;
            }
            return (
                <pre id="description" className="jsontree">
                    <JSONTree
                        hideRoot={true}
                        shouldExpandNode={() => true}  // default expandNode
                        getItemString={() => (<span />)}  // remove the {} items
                        data={openRowDataSource}
                    />
                    {
                        isLogLink
                            ?
                            <div className="logpath">
                                <span className="logName">logPath: </span>
                                <a className="logContent logHref" href={logPathRow} target="_blank">{logPathRow}</a>
                            </div>
                            :
                            <div className="logpath">
                                <span className="logName">logPath: </span>
                                <span className="logContent">{logPathRow}</span>
                            </div>
                    }
                </pre>
            );
        };

        const {
            trialProfile, searchSpace, tunerAssessor, tableData, status
        } = this.state;

        const maxRuntime = this.convertTime(trialProfile.maxDuration);
        let running;
        let runningStr = '';
        if (trialProfile.endTime === 'not over') {
            running = trialProfile.maxDuration - trialProfile.execDuration;
            runningStr = this.convertTime(running);
        } else {
            runningStr = '0';
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
                                    <span className="messcont">{maxRuntime}</span>
                                </p>
                                <p>
                                    <span>Still&nbsp;run</span>
                                    <span className="messcont">{runningStr}</span>
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
                                <p className="experStatus">
                                    Status
                                    <span className="messcont">{status}</span>
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
                                <h2>The trials that succeeded</h2>
                            </Col>
                            <Col span={6}>
                                <span className="tabuser1">top</span>
                                <Select
                                    style={{ width: 200 }}
                                    placeholder="50"
                                    optionFilterProp="children"
                                    onSelect={this.handleChange}
                                >
                                    <Option value="5">5</Option>
                                    <Option value="50">50</Option>
                                    <Option value="100">100</Option>
                                    <Option value="150">150</Option>
                                    <Option value="200">200</Option>
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
                    />
                </div>
                <div className="downExp">
                    <Button
                        type="primary"
                        className="tableButton"
                        onClick={this.downExperimentContent}
                    >
                        Download experiment summary
                    </Button>
                </div>
            </div>
        );
    }
}
export default Sessionpro;
