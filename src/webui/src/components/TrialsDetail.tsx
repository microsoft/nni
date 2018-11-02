import * as React from 'react';
import axios from 'axios';
import { MANAGER_IP } from '../static/const';
import { Row, Tabs } from 'antd';
import { TableObj, Parameters, AccurPoint } from '../static/interface';
import Accuracy from './overview/Accuracy';
import Duration from './trial-detail/Duration';
import Title1 from './overview/Title1';
import Para from './trial-detail/Para';
import TableList from './trial-detail/TableList';
const TabPane = Tabs.TabPane;
import '../static/style/trialsDetail.scss';

interface TrialDetailState {
    accSource: object;
    accNodata: string;
    tableListSource: Array<TableObj>;
}

class TrialsDetail extends React.Component<{}, TrialDetailState> {

    public _isMounted = false;
    public interAccuracy = 0;
    public interTableList = 1;
    constructor(props: {}) {
        super(props);

        this.state = {
            accSource: {},
            accNodata: '',
            tableListSource: []
        };

    }
    // trial accuracy graph
    drawPointGraph = () => {

        axios(`${MANAGER_IP}/trial-jobs`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200 && this._isMounted) {
                    const accData = res.data;
                    const accSource: Array<AccurPoint> = [];
                    Object.keys(accData).map(item => {
                        if (accData[item].status === 'SUCCEEDED' && accData[item].finalMetricData) {
                            let acc;
                            let tableAcc;
                            if (accData[item].finalMetricData) {
                                acc = JSON.parse(accData[item].finalMetricData.data);
                                if (typeof (acc) === 'object') {
                                    tableAcc = acc.default;
                                } else {
                                    tableAcc = acc;
                                }
                            }
                            accSource.push({
                                acc: tableAcc,
                                index: accData[item].sequenceId
                            });
                        }
                    });
                    const accarr: Array<number> = [];
                    const indexarr: Array<number> = [];
                    Object.keys(accSource).map(item => {
                        const items = accSource[item];
                        accarr.push(items.acc);
                        indexarr.push(items.index);
                    });
                    const allAcuracy = {
                        tooltip: {
                            trigger: 'item'
                        },
                        xAxis: {
                            name: 'Trial',
                            type: 'category',
                            data: indexarr
                        },
                        yAxis: {
                            name: 'Accuracy',
                            type: 'value',
                            data: accarr
                        },
                        series: [{
                            symbolSize: 6,
                            type: 'scatter',
                            data: accarr
                        }]
                    };

                    this.setState({ accSource: allAcuracy }, () => {
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
            });
    }

    drawTableList = () => {

        axios(`${MANAGER_IP}/trial-jobs`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    const trialJobs = res.data;
                    const trialTable: Array<TableObj> = [];
                    Object.keys(trialJobs).map(item => {
                        // only succeeded trials have finalMetricData
                        let desc: Parameters = {
                            parameters: {}
                        };
                        let acc;
                        let tableAcc = 0;
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
                            desc.parameters = JSON.parse(trialJobs[item].hyperParameters).parameters;
                        } else {
                            desc.parameters = { error: 'This trial\'s parameters are not available.' };
                        }
                        if (trialJobs[item].logPath !== undefined) {
                            desc.logPath = trialJobs[item].logPath;
                            const isHyperLink = /^http/gi.test(trialJobs[item].logPath);
                            if (isHyperLink) {
                                desc.isLink = true;
                            }
                        }
                        if (trialJobs[item].finalMetricData !== undefined) {
                            acc = JSON.parse(trialJobs[item].finalMetricData.data);
                            if (typeof (acc) === 'object') {
                                tableAcc = acc.default;
                            } else {
                                tableAcc = acc;
                            }
                        }
                        trialTable.push({
                            key: trialTable.length,
                            sequenceId: trialJobs[item].sequenceId,
                            id: id,
                            status: status,
                            duration: duration,
                            acc: tableAcc,
                            description: desc
                        });
                    });
                    if (this._isMounted) {
                        this.setState(() => ({
                            tableListSource: trialTable
                        }));
                    }
                }
            });
    }

    callback = (key: string) => {

        switch (key) {
            case '1':
                window.clearInterval(Para.intervalIDPara);
                window.clearInterval(Duration.intervalDuration);
                this.drawPointGraph();
                this.interAccuracy = window.setInterval(this.drawPointGraph, 10000);
                break;

            case '2':
                window.clearInterval(this.interAccuracy);
                window.clearInterval(Duration.intervalDuration);
                break;

            case '3':
                window.clearInterval(this.interAccuracy);
                window.clearInterval(Para.intervalIDPara);
                break;

            default:
        }
    }

    componentDidMount() {

        this._isMounted = true;
        this.drawPointGraph();
        this.drawTableList();
        this.interAccuracy = window.setInterval(this.drawPointGraph, 10000);
        this.interTableList = window.setInterval(this.drawTableList, 10000);
    }

    componentWillUnmount() {
        this._isMounted = false;
        window.clearInterval(this.interTableList);
        window.clearInterval(this.interAccuracy);
    }

    render() {
        const {
            accSource, accNodata,
            tableListSource
        } = this.state;
        const titleOfacc = (
            <Title1 text="Trial Accuracy" icon="3.png" />
        );
        const titleOfhyper = (
            <Title1 text="Hyper Parameter" icon="1.png" />
        );
        const titleOfDuration = (
            <Title1 text="Trial Duration" icon="2.png" />
        );
        return (
            <div>
                <div className="trial" id="tabsty">
                    <Tabs onChange={this.callback} type="card">
                        <TabPane tab={titleOfacc} key="1">
                            <Row className="graph">
                                <Accuracy
                                    height={432}
                                    accuracyData={accSource}
                                    accNodata={accNodata}
                                />
                            </Row>
                        </TabPane>
                        <TabPane tab={titleOfhyper} key="2">
                            <Row className="graph"><Para /></Row>
                        </TabPane>
                        <TabPane tab={titleOfDuration} key="3">
                            <Duration />
                        </TabPane>
                    </Tabs>
                </div>
                {/* trial table list */}
                <TableList
                    tableSource={tableListSource}
                    updateList={this.drawTableList}
                />
            </div>
        );
    }
}

export default TrialsDetail;