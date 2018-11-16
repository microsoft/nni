import * as React from 'react';
import axios from 'axios';
import { Row, Col, Button } from 'antd';
import { MANAGER_IP } from '../static/const';
import {
    Experiment, TableObj,
    Parameters, TrialNumber
} from '../static/interface';
import SuccessTable from './overview/SuccessTable';
import Title1 from './overview/Title1';
import Progressed from './overview/Progress';
import Accuracy from './overview/Accuracy';
import SearchSpace from './overview/SearchSpace';
import BasicInfo from './overview/BasicInfo';
import TrialPro from './overview/TrialProfile';

require('../static/style/overview.scss');
require('../static/style/logPath.scss');
require('../static/style/accuracy.css');
require('../static/style/table.scss');
require('../static/style/overviewTitle.scss');

interface OverviewState {
    tableData: Array<TableObj>;
    searchSpace: object;
    status: string;
    errorStr: string;
    trialProfile: Experiment;
    option: object;
    noData: string;
    accuracyData: object;
    bestAccuracy: string;
    accNodata: string;
    trialNumber: TrialNumber;
    downBool: boolean;
}

class Overview extends React.Component<{}, OverviewState> {

    public _isMounted = false;
    public intervalID = 0;
    public intervalProfile = 1;

    constructor(props: {}) {
        super(props);
        this.state = {
            searchSpace: {},
            status: '',
            errorStr: '',
            trialProfile: {
                id: '',
                author: '',
                experName: '',
                runConcurren: 0,
                maxDuration: 0,
                execDuration: 0,
                MaxTrialNum: 0,
                startTime: 0,
                tuner: {},
                trainingServicePlatform: ''
            },
            tableData: [{
                key: 0,
                sequenceId: 0,
                id: '',
                duration: 0,
                status: '',
                acc: 0,
                description: {
                    parameters: {}
                }
            }],
            option: {},
            noData: '',
            // accuracy
            accuracyData: {},
            accNodata: '',
            bestAccuracy: '',
            trialNumber: {
                succTrial: 0,
                failTrial: 0,
                stopTrial: 0,
                waitTrial: 0,
                runTrial: 0,
                unknowTrial: 0,
                totalCurrentTrial: 0
            },
            downBool: false
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
                    let trialPro = [];
                    const trainingPlatform = sessionData.params.trainingServicePlatform;
                    // assessor clusterMeteData
                    const clusterMetaData = sessionData.params.clusterMetaData;
                    const endTimenum = sessionData.endTime;
                    const assessor = sessionData.params.assessor;
                    const advisor = sessionData.params.advisor;
                    trialPro.push({
                        id: sessionData.id,
                        author: sessionData.params.authorName,
                        revision: sessionData.revision,
                        experName: sessionData.params.experimentName,
                        runConcurren: sessionData.params.trialConcurrency,
                        logDir: sessionData.logDir ? sessionData.logDir : 'undefined',
                        maxDuration: sessionData.params.maxExecDuration,
                        execDuration: sessionData.execDuration,
                        MaxTrialNum: sessionData.params.maxTrialNum,
                        startTime: sessionData.startTime,
                        endTime: endTimenum ? endTimenum : undefined,
                        trainingServicePlatform: trainingPlatform,
                        tuner: sessionData.params.tuner,
                        assessor: assessor ? assessor : undefined,
                        advisor: advisor ? advisor : undefined,
                        clusterMetaData: clusterMetaData ? clusterMetaData : undefined
                    });
                    // search space format loguniform max and min
                    const searchSpace = JSON.parse(sessionData.params.searchSpace);
                    Object.keys(searchSpace).map(item => {
                        const key = searchSpace[item]._type;
                        let value = searchSpace[item]._value;
                        switch (key) {
                            case 'loguniform':
                            case 'qloguniform':
                                const a = Math.pow(Math.E, value[0]);
                                const b = Math.pow(Math.E, value[1]);
                                value = [a, b];
                                searchSpace[item]._value = value;
                                break;

                            case 'quniform':
                            case 'qnormal':
                            case 'qlognormal':
                                searchSpace[item]._value = [value[0], value[1]];
                                break;

                            default:

                        }
                    });
                    if (this._isMounted) {
                        this.setState({
                            trialProfile: trialPro[0],
                            searchSpace: searchSpace
                        });
                    }
                }
            });
        this.checkStatus();

    }

    checkStatus = () => {
        axios(`${MANAGER_IP}/check-status`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200 && this._isMounted) {
                    const errors = res.data.errors;
                    if (errors.length !== 0) {
                        this.setState({
                            status: res.data.status,
                            errorStr: res.data.errors[0]
                        });
                    } else {
                        this.setState({
                            status: res.data.status,
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
                    const tableData = res.data;
                    const topTableData: Array<TableObj> = [];
                    const profile: TrialNumber = {
                        succTrial: 0,
                        failTrial: 0,
                        stopTrial: 0,
                        waitTrial: 0,
                        runTrial: 0,
                        unknowTrial: 0,
                        totalCurrentTrial: 0
                    };
                    // currently totoal number
                    profile.totalCurrentTrial = tableData.length;
                    Object.keys(tableData).map(item => {
                        switch (tableData[item].status) {
                            case 'WAITING':
                                profile.waitTrial += 1;
                                break;

                            case 'UNKNOWN':
                                profile.unknowTrial += 1;
                                break;

                            case 'FAILED':
                                profile.failTrial += 1;
                                break;

                            case 'USER_CANCELED':
                            case 'SYS_CANCELED':
                                profile.stopTrial += 1;
                                break;
                            case 'SUCCEEDED':
                                profile.succTrial += 1;
                                const desJobDetail: Parameters = {
                                    parameters: {}
                                };
                                const duration = (tableData[item].endTime - tableData[item].startTime) / 1000;
                                let acc;
                                let tableAcc = 0;
                                if (tableData[item].finalMetricData) {
                                    acc = JSON.parse(tableData[item].finalMetricData.data);
                                    if (typeof (acc) === 'object') {
                                        if (acc.default) {
                                            tableAcc = acc.default;
                                        }
                                    } else {
                                        tableAcc = acc;
                                    }
                                }
                                // if hyperparameters is undefine, show error message, else, show parameters value
                                if (tableData[item].hyperParameters) {
                                    desJobDetail.parameters = JSON.parse(tableData[item].hyperParameters).parameters;
                                } else {
                                    desJobDetail.parameters = { error: 'This trial\'s parameters are not available.' };
                                }
                                if (tableData[item].logPath !== undefined) {
                                    desJobDetail.logPath = tableData[item].logPath;
                                    const isSessionLink = /^http/gi.test(tableData[item].logPath);
                                    if (isSessionLink) {
                                        desJobDetail.isLink = true;
                                    }
                                }
                                topTableData.push({
                                    key: topTableData.length,
                                    sequenceId: tableData[item].sequenceId,
                                    id: tableData[item].id,
                                    duration: duration,
                                    status: tableData[item].status,
                                    acc: tableAcc,
                                    description: desJobDetail
                                });
                                break;
                            default:
                        }
                    });
                    topTableData.sort((a: TableObj, b: TableObj) => {
                        if (a.acc && b.acc) {
                            return b.acc - a.acc;
                        } else {
                            return NaN;
                        }
                    });
                    topTableData.length = Math.min(10, topTableData.length);
                    if (this._isMounted) {
                        this.setState({
                            tableData: topTableData,
                            trialNumber: profile
                        });
                    }
                    // draw accuracy
                    this.drawPointGraph();
                }
            });
    }

    downExperimentContent = () => {
        this.setState(() => ({
            downBool: true
        }));
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
                    const isEdge = navigator.userAgent.indexOf('Edge') !== -1 ? true : false;
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
                    aTag.download = 'experiment.json';
                    aTag.href = URL.createObjectURL(file);
                    aTag.click();
                    if (!isEdge) {
                        URL.revokeObjectURL(aTag.href);
                    }
                    if (navigator.userAgent.indexOf('Firefox') > -1) {
                        const downTag = document.createElement('a');
                        downTag.addEventListener('click', function () {
                            downTag.download = 'experiment.json';
                            downTag.href = URL.createObjectURL(file);
                        });
                        let eventMouse = document.createEvent('MouseEvents');
                        eventMouse.initEvent('click', false, false);
                        downTag.dispatchEvent(eventMouse);
                    }
                    this.setState(() => ({
                        downBool: false
                    }));
                }
            }));
    }

    // trial accuracy graph Default Metric
    drawPointGraph = () => {

        const { tableData } = this.state;
        const sourcePoint = JSON.parse(JSON.stringify(tableData));
        sourcePoint.sort((a: TableObj, b: TableObj) => {
            if (a.sequenceId && b.sequenceId) {
                return a.sequenceId - b.sequenceId;
            } else {
                return NaN;
            }
        });
        const accarr: Array<number> = [];
        const indexarr: Array<number> = [];
        Object.keys(sourcePoint).map(item => {
            const items = sourcePoint[item];
            accarr.push(items.acc);
            indexarr.push(items.sequenceId);
        });
        const bestAccnum = Math.max(...accarr);
        const accOption = {
            tooltip: {
                trigger: 'item'
            },
            xAxis: {
                name: 'Trial',
                type: 'category',
                data: indexarr
            },
            yAxis: {
                name: 'Default Metric',
                type: 'value',
                data: accarr
            },
            series: [{
                symbolSize: 6,
                type: 'scatter',
                data: accarr
            }]
        };
        this.setState({ accuracyData: accOption }, () => {
            if (accarr.length === 0) {
                this.setState({
                    accNodata: 'No data'
                });
            } else {
                this.setState({
                    accNodata: '',
                    bestAccuracy: bestAccnum.toFixed(6)
                });
            }
        });
    }

    componentDidMount() {
        this._isMounted = true;
        this.showSessionPro();
        this.showTrials();
        this.intervalID = window.setInterval(this.showTrials, 10000);
        this.intervalProfile = window.setInterval(this.showSessionPro, 60000);
    }

    componentWillUnmount() {
        this._isMounted = false;
        window.clearInterval(this.intervalID);
        window.clearInterval(this.intervalProfile);
    }

    render() {

        const {
            trialProfile,
            searchSpace,
            tableData,
            accuracyData,
            accNodata,
            status,
            errorStr,
            trialNumber,
            bestAccuracy,
            downBool
        } = this.state;

        return (
            <div className="overview">
                {/* status and experiment block */}
                <Row>
                    <Row className="exbgcolor">
                        <Col span={4}><Title1 text="Experiment" icon="11.png" /></Col>
                        <Col span={4}>
                            <Button
                                type="primary"
                                className="changeBtu download"
                                onClick={this.downExperimentContent}
                                disabled={downBool}
                            >
                                <span>Download</span>
                                <img src={require('../static/img/icon/download.png')} alt="icon" />
                            </Button>
                        </Col>
                    </Row>
                    <BasicInfo trialProfile={trialProfile} status={status} />
                </Row>
                <Row className="overMessage">
                    {/* status graph */}
                    <Col span={8} className="prograph overviewBoder">
                        <Title1 text="Status" icon="5.png" />
                        <Progressed
                            trialNumber={trialNumber}
                            trialProfile={trialProfile}
                            bestAccuracy={bestAccuracy}
                            status={status}
                            errors={errorStr}
                        />
                    </Col>
                    {/* experiment parameters search space tuner assessor... */}
                    <Col span={8} className="overviewBoder">
                        <Title1 text="Search Space" icon="10.png" />
                        <Row className="experiment">
                            <SearchSpace searchSpace={searchSpace} />
                        </Row>
                    </Col>
                    <Col span={8} className="overviewBoder">
                        <Title1 text="Trial Profile" icon="4.png" />
                        <Row className="experiment">
                            {/* the scroll bar all the trial profile in the searchSpace div*/}
                            <div className="experiment searchSpace">
                                <TrialPro
                                    tiralProInfo={trialProfile}
                                />
                            </div>
                        </Row>
                    </Col>
                </Row>
                <Row className="overGraph">
                    <Col span={8} className="overviewBoder">
                        <Title1 text="Optimization Progress" icon="3.png" />
                        <Row className="accuracy">
                            <Accuracy
                                accuracyData={accuracyData}
                                accNodata={accNodata}
                                height={324}
                            />
                        </Row>
                    </Col>
                    <Col span={16} id="succeTable">
                        <Title1 text="Top10 Trials" icon="7.png" />
                        <SuccessTable tableSource={tableData} />
                    </Col>
                </Row>
            </div>
        );
    }
}
export default Overview;
