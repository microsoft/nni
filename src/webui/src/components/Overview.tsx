import * as React from 'react';
import axios from 'axios';
import { Row, Col } from 'antd';
import { MANAGER_IP } from '../static/const';
import { Experiment, TableObj, Parameters, TrialNumber } from '../static/interface';
import { getFinal } from '../static/function';
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
    experimentAPI: object;
    searchSpace: object;
    status: string;
    errorStr: string;
    trialProfile: Experiment;
    option: object;
    noData: string;
    accuracyData: object;
    bestAccuracy: number;
    accNodata: string;
    trialNumber: TrialNumber;
    isTop10: boolean;
    titleMaxbgcolor?: string;
    titleMinbgcolor?: string;
    // trial stdout is content(false) or link(true)
    isLogCollection: boolean;
    isMultiPhase: boolean;
}

class Overview extends React.Component<{}, OverviewState> {

    public _isMounted = false;
    public intervalID = 0;
    public intervalProfile = 1;

    constructor(props: {}) {
        super(props);
        this.state = {
            searchSpace: {},
            experimentAPI: {},
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
            tableData: [],
            option: {},
            noData: '',
            // accuracy
            accuracyData: {},
            accNodata: '',
            bestAccuracy: 0,
            trialNumber: {
                succTrial: 0,
                failTrial: 0,
                stopTrial: 0,
                waitTrial: 0,
                runTrial: 0,
                unknowTrial: 0,
                totalCurrentTrial: 0
            },
            isTop10: true,
            isLogCollection: false,
            isMultiPhase: false
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
                    const tempara = sessionData.params;
                    const trainingPlatform = tempara.trainingServicePlatform;
                    // assessor clusterMeteData
                    const clusterMetaData = tempara.clusterMetaData;
                    const endTimenum = sessionData.endTime;
                    const assessor = tempara.assessor;
                    const advisor = tempara.advisor;
                    let optimizeMode = 'other';
                    if (tempara.tuner !== undefined) {
                        if (tempara.tuner.classArgs !== undefined) {
                            if (tempara.tuner.classArgs.optimize_mode !== undefined) {
                                optimizeMode = tempara.tuner.classArgs.optimize_mode;
                            }
                        }
                    }
                    // default logCollection is true
                    const logCollection = tempara.logCollection;
                    let expLogCollection: boolean = false;
                    const isMultiy: boolean = tempara.multiPhase !== undefined
                        ? tempara.multiPhase : false;
                    if (optimizeMode !== undefined) {
                        if (optimizeMode === 'minimize') {
                            if (this._isMounted) {
                                this.setState({
                                    isTop10: false,
                                    titleMinbgcolor: '#999'
                                });
                            }
                        } else {
                            if (this._isMounted) {
                                this.setState({
                                    isTop10: true,
                                    titleMaxbgcolor: '#999'
                                });
                            }
                        }
                    }
                    if (logCollection !== undefined && logCollection !== 'none') {
                        expLogCollection = true;
                    }
                    trialPro.push({
                        id: sessionData.id,
                        author: tempara.authorName,
                        revision: sessionData.revision,
                        experName: tempara.experimentName,
                        runConcurren: tempara.trialConcurrency,
                        logDir: sessionData.logDir ? sessionData.logDir : 'undefined',
                        maxDuration: tempara.maxExecDuration,
                        execDuration: sessionData.execDuration,
                        MaxTrialNum: tempara.maxTrialNum,
                        startTime: sessionData.startTime,
                        endTime: endTimenum ? endTimenum : undefined,
                        trainingServicePlatform: trainingPlatform,
                        tuner: tempara.tuner,
                        assessor: assessor ? assessor : undefined,
                        advisor: advisor ? advisor : undefined,
                        clusterMetaData: clusterMetaData ? clusterMetaData : undefined,
                        logCollection: logCollection
                    });
                    // search space format loguniform max and min
                    const temp = tempara.searchSpace;
                    const searchSpace = temp !== undefined
                        ? JSON.parse(temp) : {};
                    Object.keys(searchSpace).map(item => {
                        const key = searchSpace[item]._type;
                        let value = searchSpace[item]._value;
                        switch (key) {
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
                            experimentAPI: res.data,
                            trialProfile: trialPro[0],
                            searchSpace: searchSpace,
                            isLogCollection: expLogCollection,
                            isMultiPhase: isMultiy
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
                if (res.status === 200) {
                    const errors = res.data.errors;
                    if (errors.length !== 0) {
                        if (this._isMounted) {
                            this.setState({
                                status: res.data.status,
                                errorStr: res.data.errors[0]
                            });
                        }
                    } else {
                        if (this._isMounted) {
                            this.setState({
                                status: res.data.status,
                            });
                        }
                    }
                }
            });
    }

    showTrials = () => {
        this.isOffInterval();
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

                            case 'RUNNING':
                                profile.runTrial += 1;
                                break;

                            case 'USER_CANCELED':
                            case 'SYS_CANCELED':
                            case 'EARLY_STOPPED':
                                profile.stopTrial += 1;
                                break;
                            case 'SUCCEEDED':
                                profile.succTrial += 1;
                                const desJobDetail: Parameters = {
                                    parameters: {},
                                    intermediate: []
                                };
                                const duration = (tableData[item].endTime - tableData[item].startTime) / 1000;
                                const acc = getFinal(tableData[item].finalMetricData);
                                // if hyperparameters is undefine, show error message, else, show parameters value
                                const tempara = tableData[item].hyperParameters;
                                if (tempara !== undefined) {
                                    const tempLength = tempara.length;
                                    const parameters = JSON.parse(tempara[tempLength - 1]).parameters;
                                    if (typeof parameters === 'string') {
                                        desJobDetail.parameters = JSON.parse(parameters);
                                    } else {
                                        desJobDetail.parameters = parameters;
                                    }
                                } else {
                                    desJobDetail.parameters = { error: 'This trial\'s parameters are not available.' };
                                }
                                if (tableData[item].logPath !== undefined) {
                                    desJobDetail.logPath = tableData[item].logPath;
                                }
                                topTableData.push({
                                    key: topTableData.length,
                                    sequenceId: tableData[item].sequenceId,
                                    id: tableData[item].id,
                                    duration: duration,
                                    status: tableData[item].status,
                                    acc: acc,
                                    description: desJobDetail
                                });
                                break;
                            default:
                        }
                    });
                    // choose top10 or lowest10
                    const { isTop10 } = this.state;
                    if (isTop10 === true) {
                        topTableData.sort((a: TableObj, b: TableObj) => {
                            if (a.acc !== undefined && b.acc !== undefined) {
                                return JSON.parse(b.acc.default) - JSON.parse(a.acc.default);
                            } else {
                                return NaN;
                            }
                        });
                    } else {
                        topTableData.sort((a: TableObj, b: TableObj) => {
                            if (a.acc !== undefined && b.acc !== undefined) {
                                return JSON.parse(a.acc.default) - JSON.parse(b.acc.default);
                            } else {
                                return NaN;
                            }
                        });
                    }
                    topTableData.length = Math.min(10, topTableData.length);
                    let bestDefaultMetric = 0;
                    if (topTableData[0] !== undefined) {
                        if (topTableData[0].acc !== undefined) {
                            bestDefaultMetric = JSON.parse(topTableData[0].acc.default);
                        }
                    }
                    if (this._isMounted) {
                        this.setState({
                            tableData: topTableData,
                            trialNumber: profile,
                            bestAccuracy: bestDefaultMetric
                        });
                    }
                    this.checkStatus();
                    // draw accuracy
                    this.drawPointGraph();
                }
            });
    }

    // trial accuracy graph Default Metric
    drawPointGraph = () => {

        const { tableData } = this.state;
        const sourcePoint = JSON.parse(JSON.stringify(tableData));
        sourcePoint.sort((a: TableObj, b: TableObj) => {
            if (a.sequenceId !== undefined && b.sequenceId !== undefined) {
                return a.sequenceId - b.sequenceId;
            } else {
                return NaN;
            }
        });
        const accarr: Array<number> = [];
        const indexarr: Array<number> = [];
        Object.keys(sourcePoint).map(item => {
            const items = sourcePoint[item];
            accarr.push(items.acc.default);
            indexarr.push(items.sequenceId);
        });
        const accOption = {
            // support max show 0.0000000
            grid: {
                left: 67,
                right: 40
            },
            tooltip: {
                trigger: 'item'
            },
            xAxis: {
                name: 'Trial',
                type: 'category',
                data: indexarr
            },
            yAxis: {
                name: 'Default metric',
                type: 'value',
                scale: true,
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
                    accNodata: ''
                });
            }
        });
    }

    clickMaxTop = (event: React.SyntheticEvent<EventTarget>) => {
        event.stopPropagation();
        // #999 panel active bgcolor; #b3b3b3 as usual
        this.setState(() => ({ isTop10: true, titleMaxbgcolor: '#999', titleMinbgcolor: '#b3b3b3' }));
        this.showTrials();
    }

    clickMinTop = (event: React.SyntheticEvent<EventTarget>) => {
        event.stopPropagation();
        this.setState(() => ({ isTop10: false, titleMaxbgcolor: '#b3b3b3', titleMinbgcolor: '#999' }));
        this.showTrials();
    }

    isOffInterval = () => {
        const { status } = this.state;
        switch (status) {
            case 'DONE':
            case 'ERROR':
            case 'STOPPED':
                window.clearInterval(this.intervalID);
                window.clearInterval(this.intervalProfile);
                break;
            default:
        }
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
            trialProfile, searchSpace, tableData, accuracyData,
            accNodata, status, errorStr, trialNumber, bestAccuracy, isMultiPhase,
            titleMaxbgcolor, titleMinbgcolor, isLogCollection, experimentAPI
        } = this.state;

        return (
            <div className="overview">
                {/* status and experiment block */}
                <Row>
                    <Title1 text="Experiment" icon="11.png" />
                    <BasicInfo trialProfile={trialProfile} status={status} />
                </Row>
                <Row className="overMessage">
                    {/* status graph */}
                    <Col span={9} className="prograph overviewBoder">
                        <Title1 text="Status" icon="5.png" />
                        <Progressed
                            trialNumber={trialNumber}
                            trialProfile={trialProfile}
                            bestAccuracy={bestAccuracy}
                            status={status}
                            errors={errorStr}
                            updateFile={this.showSessionPro}
                        />
                    </Col>
                    {/* experiment parameters search space tuner assessor... */}
                    <Col span={7} className="overviewBoder">
                        <Title1 text="Search space" icon="10.png" />
                        <Row className="experiment">
                            <SearchSpace searchSpace={searchSpace} />
                        </Row>
                    </Col>
                    <Col span={8} className="overviewBoder">
                        <Title1 text="Profile" icon="4.png" />
                        <Row className="experiment">
                            {/* the scroll bar all the trial profile in the searchSpace div*/}
                            <div className="experiment searchSpace">
                                <TrialPro experiment={experimentAPI} />
                            </div>
                        </Row>
                    </Col>
                </Row>
                <Row className="overGraph">
                    <Row className="top10bg">
                        <Col span={4} className="top10Title">
                            <Title1 text="Top10  trials" icon="7.png" />
                        </Col>
                        <Col
                            span={2}
                            className="title"
                            onClick={this.clickMaxTop}
                        >
                            <Title1 text="Maximal" icon="max.png" bgcolor={titleMaxbgcolor} />
                        </Col>
                        <Col
                            span={2}
                            className="title minTitle"
                            onClick={this.clickMinTop}
                        >
                            <Title1 text="Minimal" icon="min.png" bgcolor={titleMinbgcolor} />
                        </Col>
                    </Row>
                    <Row>
                        <Col span={8} className="overviewBoder">
                            <Row className="accuracy">
                                <Accuracy
                                    accuracyData={accuracyData}
                                    accNodata={accNodata}
                                    height={324}
                                />
                            </Row>
                        </Col>
                        <Col span={16} id="succeTable">
                            <SuccessTable
                                tableSource={tableData}
                                multiphase={isMultiPhase}
                                logCollection={isLogCollection}
                                trainingPlatform={trialProfile.trainingServicePlatform}
                            />
                        </Col>
                    </Row>
                </Row>
            </div>
        );
    }
}
export default Overview;
