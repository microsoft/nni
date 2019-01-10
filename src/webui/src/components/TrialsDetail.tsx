import * as React from 'react';
import axios from 'axios';
import { MANAGER_IP } from '../static/const';
import { Row, Col, Tabs, Input, Select, Button } from 'antd';
const Option = Select.Option;
import { TableObjFianl, Parameters, DetailAccurPoint, TooltipForAccuracy } from '../static/interface';
import { getFinalResult, getFinal } from '../static/function';
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
    tableListSource: Array<TableObjFianl>;
    searchResultSource: Array<TableObjFianl>;
    isHasSearch: boolean;
    experimentStatus: string;
    entriesTable: number;
    experimentPlatform: string;
}

class TrialsDetail extends React.Component<{}, TrialDetailState> {

    public _isMounted = false;
    public interAccuracy = 0;
    public interTableList = 1;
    public interAllTableList = 2;

    public tableList: TableList | null;

    constructor(props: {}) {
        super(props);

        this.state = {
            accSource: {},
            accNodata: '',
            tableListSource: [],
            searchResultSource: [],
            experimentStatus: '',
            entriesTable: 20,
            isHasSearch: false,
            experimentPlatform: ''
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
                            let searchSpace: object = {};
                            const acc = getFinalResult(accData[item].finalMetricData);
                            if (accData[item].hyperParameters) {
                                searchSpace = JSON.parse(accData[item].hyperParameters).parameters;
                            }
                            accSource.push({
                                acc: acc,
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
                                    return [point[0], 80];
                                } else {
                                    return [point[0] - 300, 80];
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
        this.isOffIntervals();
        axios.get(`${MANAGER_IP}/trial-jobs`)
            .then(res => {
                if (res.status === 200) {
                    const trialJobs = res.data;
                    const trialTable: Array<TableObjFianl> = [];
                    Object.keys(trialJobs).map(item => {
                        // only succeeded trials have finalMetricData
                        let desc: Parameters = {
                            parameters: {}
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
                    // search part data
                    const { searchResultSource } = this.state;
                    if (searchResultSource.length !== 0) {
                        const temp: Array<number> = [];
                        Object.keys(searchResultSource).map(index => {
                            temp.push(searchResultSource[index].id);
                        });
                        const searchResultList: Array<TableObjFianl> = [];
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
            });
    }

    // update all data in table
    drawAllTableList = () => {
        this.isOffIntervals();
        axios.get(`${MANAGER_IP}/trial-jobs`)
            .then(res => {
                if (res.status === 200) {
                    const trialJobs = res.data;
                    const trialTable: Array<TableObjFianl> = [];
                    Object.keys(trialJobs).map(item => {
                        // only succeeded trials have finalMetricData
                        let desc: Parameters = {
                            parameters: {}
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
                    if (this._isMounted) {
                        this.setState(() => ({
                            tableListSource: trialTable,
                            searchResultSource: trialTable
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
                this.isOffIntervals();
                window.clearInterval(this.interAccuracy);
                window.clearInterval(Duration.intervalDuration);
                break;

            case '3':
                this.isOffIntervals();
                window.clearInterval(this.interAccuracy);
                window.clearInterval(Para.intervalIDPara);
                break;

            default:
        }
    }

    // search a trial by trial No. & trial id
    searchTrial = (event: React.ChangeEvent<HTMLInputElement>) => {
        const targetValue = event.target.value;
        if (targetValue === '' || targetValue === ' ') {
            this.drawAllTableList();
            this.interAllTableList = window.setInterval(this.drawAllTableList, 10000);
        } else {
            window.clearInterval(this.interAllTableList);
            const { tableListSource } = this.state;
            const searchResultList: Array<TableObjFianl> = [];
            Object.keys(tableListSource).map(key => {
                const item = tableListSource[key];
                if (item.sequenceId.toString() === targetValue || item.id.includes(targetValue)) {
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
                            window.clearInterval(this.interAccuracy);
                            window.clearInterval(this.interTableList);
                            window.clearInterval(Duration.intervalDuration);
                            window.clearInterval(Para.intervalIDPara);
                            window.clearInterval(this.interAllTableList);
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
                this.setState(() => ({ entriesTable: 100000 }));
                break;
            default:
        }
    }

    test = () => {
        alert('TableList component was not properly initialized.');
    }

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
                    if (this._isMounted) {
                        this.setState({
                            experimentPlatform: trainingPlatform
                        });
                    }
                }
            });
    }

    componentDidMount() {

        this._isMounted = true;
        this.drawTableList();
        this.drawPointGraph();
        this.interTableList = window.setInterval(this.drawTableList, 10000);
        this.interAccuracy = window.setInterval(this.drawPointGraph, 10000);
        this.checkExperimentPlatform();
    }

    componentWillUnmount() {
        this._isMounted = false;
        window.clearInterval(this.interTableList);
        window.clearInterval(this.interAccuracy);
    }

    render() {
        const { accSource, accNodata, tableListSource, 
            entriesTable, searchResultSource, isHasSearch,
            experimentPlatform
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
                <Title1 text="All Trials" icon="6.png" />
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
                                {/* <span>Search:</span> */}
                                <Input
                                    type="text"
                                    placeholder="search by Trial No. and id"
                                    onChange={this.searchTrial}
                                    style={{ width: 200, marginLeft: 6 }}
                                />
                            </Col>
                        </Row>
                    </Col>
                </Row>
                <TableList
                    entries={entriesTable}
                    tableSource={tableListSource}
                    updateList={this.drawTableList}
                    searchResult={searchResultSource}
                    isHasSearch={isHasSearch}
                    platform={experimentPlatform}
                    ref={(tabList) => this.tableList = tabList}
                />
            </div>
        );
    }
}

export default TrialsDetail;