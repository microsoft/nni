import * as React from 'react';
import axios from 'axios';
import { MANAGER_IP } from '../static/const';
import { Row, Col, Button, Tabs, Input } from 'antd';
const Search = Input.Search;
import { TableObj, Parameters, DetailAccurPoint, TooltipForAccuracy } from '../static/interface';
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
    tableBaseSource: Array<TableObj>;
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
            tableListSource: [],
            tableBaseSource: []
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
                    const accSource: Array<DetailAccurPoint> = [];
                    Object.keys(accData).map(item => {
                        if (accData[item].status === 'SUCCEEDED' && accData[item].finalMetricData) {
                            let acc;
                            let tableAcc;
                            let searchSpace: object = {};
                            if (accData[item].finalMetricData) {
                                acc = JSON.parse(accData[item].finalMetricData.data);
                                if (typeof (acc) === 'object') {
                                    if (acc.default) {
                                        tableAcc = acc.default;
                                    }
                                } else {
                                    tableAcc = acc;
                                }
                            }
                            if (accData[item].hyperParameters) {
                                searchSpace = JSON.parse(accData[item].hyperParameters).parameters;
                            }
                            accSource.push({
                                acc: tableAcc,
                                index: accData[item].sequenceId,
                                searchSpace: JSON.stringify(searchSpace)
                            });
                        }
                    });
                    const resultList: Array<number | string>[] = [];
                    Object.keys(accSource).map(item => {
                        const items = accSource[item];
                        let temp: Array<number | string>;
                        temp = [items.index, items.acc, JSON.parse(items.searchSpace)];
                        resultList.push(temp);
                    });
                    const allAcuracy = {
                        tooltip: {
                            trigger: 'item',
                            enterable: true,
                            position: function (point: Array<number>, data: TooltipForAccuracy) {
                                if (data.data[0] < resultList.length / 2) {
                                    return [point[0], 10];
                                } else {
                                    return [point[0] - 300, 10];
                                }
                            },
                            formatter: function (data: TooltipForAccuracy) {
                                const result = '<div class="tooldetailAccuracy">' +
                                    '<div>Trial No: ' + data.data[0] + '</div>' +
                                    '<div>Default Metrc: ' + data.data[1] + '</div>' +
                                    '<div>Parameters: ' +
                                    '<pre>' + JSON.stringify(data.data[2], null, 4) + '</pre>' +
                                    '</div>' +
                                    '</div>';
                                return result;
                            }
                        },
                        xAxis: {
                            name: 'Trial',
                            type: 'category',
                        },
                        yAxis: {
                            name: 'Default Metric',
                            type: 'value',
                        },
                        series: [{
                            symbolSize: 6,
                            type: 'scatter',
                            data: resultList
                        }]
                    };

                    this.setState({ accSource: allAcuracy }, () => {
                        if (resultList.length === 0) {
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
                        // only succeeded trials have finalMetricData
                        let desc: Parameters = {
                            parameters: {},
                            intermediate: []
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
                        let mediate: Array<string> = [];
                        Object.keys(metricSource).map(key => {
                            const items = metricSource[key];
                            if (items.trialJobId === id) {
                                mediate.push(items.data);
                            }
                        });
                        desc.intermediate = mediate;
                        if (trialJobs[item].finalMetricData !== undefined) {
                            acc = JSON.parse(trialJobs[item].finalMetricData.data);
                            if (typeof (acc) === 'object') {
                                if (acc.default) {
                                    tableAcc = acc.default;
                                }
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
                            tableListSource: trialTable,
                            tableBaseSource: trialTable
                        }));
                    }
                }
            }));
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

    // search a specific trial by trial No.
    searchTrialNo = (value: string) => {

        window.clearInterval(this.interTableList);
        const { tableBaseSource } = this.state;
        const searchResultList: Array<TableObj> = [];
        Object.keys(tableBaseSource).map(key => {
            const item = tableBaseSource[key];
            if (item.sequenceId.toString() === value) {
                searchResultList.push(item);
            }
        });
        this.setState(() => ({
            tableListSource: searchResultList
        }));
    }

    // reset btn click: rerender table
    resetRenderTable = () => {

        const searchInput = document.getElementById('searchTrial') as HTMLInputElement;
        if (searchInput !== null) {
            searchInput.value = '';
        }
        this.drawTableList();
        this.interTableList = window.setInterval(this.drawTableList, 10000);
    }
    componentDidMount() {

        this._isMounted = true;
        this.drawTableList();
        this.drawPointGraph();
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
            <Title1 text="Default Metric" icon="3.png" />
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
                <Row className="allList">
                    <Col span={12}>
                        <Title1 text="All Trials" icon="6.png" />
                    </Col>
                    <Col span={12} className="btns">
                        <Search
                            placeholder="input search Trial No."
                            onSearch={value => this.searchTrialNo(value)}
                            style={{ width: 200 }}
                            id="searchTrial"
                        />
                        <Button
                            type="primary"
                            className="tableButton resetBtn"
                            onClick={this.resetRenderTable}
                        >
                            Reset
                        </Button>
                    </Col>
                </Row>
                <TableList
                    tableSource={tableListSource}
                    updateList={this.drawTableList}
                />
            </div>
        );
    }
}

export default TrialsDetail;