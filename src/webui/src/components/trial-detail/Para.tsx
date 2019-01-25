import * as React from 'react';
import axios from 'axios';
import { MANAGER_IP } from '../../static/const';
import ReactEcharts from 'echarts-for-react';
import { Row, Col, Select, Button, message } from 'antd';
import { ParaObj, VisualMapValue, Dimobj } from '../../static/interface';
import { getFinalResult } from '../../static/function';
const Option = Select.Option;
require('echarts/lib/chart/parallel');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');
require('echarts/lib/component/visualMap');
require('../../static/style/para.scss');
require('../../static/style/button.scss');

interface ParaState {
    option: object;
    paraBack: ParaObj;
    dimName: Array<string>;
    swapAxisArr: Array<string>;
    percent: number;
    paraNodata: string;
    visualValue: VisualMapValue;
}

interface SearchSpace {
    _value: Array<number | string>;
    _type: string;
}

message.config({
    top: 250,
    duration: 2,
});

class Para extends React.Component<{}, ParaState> {

    static intervalIDPara = 4;
    public _isMounted = false;

    constructor(props: {}) {
        super(props);
        this.state = {
            option: {},
            dimName: [],
            paraBack: {
                parallelAxis: [{
                    dim: 0,
                    name: ''
                }],
                data: []
            },
            swapAxisArr: [],
            percent: 0,
            paraNodata: '',
            visualValue: {
                minAccuracy: 0,
                maxAccuracy: 1
            }
        };
    }

    getParallelAxis =
        (
            dimName: Array<string>, searchRange: SearchSpace,
            parallelAxis: Array<Dimobj>, accPara: Array<number>,
            eachTrialParams: Array<string>, paraYdata: number[][]
        ) => {
            if (this._isMounted) {
                this.setState(() => ({
                    dimName: dimName,
                    visualValue: {
                        minAccuracy: accPara.length !== 0 ? Math.min(...accPara) : 0,
                        maxAccuracy: accPara.length !== 0 ? Math.max(...accPara) : 1
                    }
                }));
            }
            // search space range and specific value [only number]
            for (let i = 0; i < dimName.length; i++) {
                const searchKey = searchRange[dimName[i]];
                switch (searchKey._type) {
                    case 'uniform':
                    case 'quniform':
                        parallelAxis.push({
                            dim: i,
                            name: dimName[i],
                            max: searchKey._value[1],
                            min: searchKey._value[0]
                        });
                        break;

                    case 'randint':
                        parallelAxis.push({
                            dim: i,
                            name: dimName[i],
                            max: searchKey._value[0],
                            min: 0
                        });
                        break;

                    case 'choice':
                        const data: Array<string> = [];
                        for (let j = 0; j < searchKey._value.length; j++) {
                            data.push(searchKey._value[j].toString());
                        }
                        parallelAxis.push({
                            dim: i,
                            name: dimName[i],
                            type: 'category',
                            data: data
                        });
                        break;
                    // support log distribute
                    case 'loguniform':
                        parallelAxis.push({
                            dim: i,
                            name: dimName[i],
                            type: 'log',
                        });
                        break;

                    default:
                        parallelAxis.push({
                            dim: i,
                            name: dimName[i]
                        });

                }
            }
            // get data for every lines. if dim is choice type, number -> toString()
            Object.keys(eachTrialParams).map(item => {
                let temp: Array<number> = [];
                for (let i = 0; i < dimName.length; i++) {
                    if ('type' in parallelAxis[i]) {
                        temp.push(
                            eachTrialParams[item][dimName[i]].toString()
                        );
                    } else {
                        temp.push(
                            eachTrialParams[item][dimName[i]]
                        );
                    }
                }
                paraYdata.push(temp);
            });
        }

    hyperParaPic = () => {
        axios
            .all([
                axios.get(`${MANAGER_IP}/trial-jobs`),
                axios.get(`${MANAGER_IP}/experiment`)
            ])
            .then(axios.spread((res, res1) => {
                if (res.status === 200 && res1.status === 200) {
                    if (res.data.length !== 0) {
                        const accParaData = res.data;
                        const accPara: Array<number> = [];
                        // specific value array
                        const eachTrialParams: Array<string> = [];
                        const parallelAxis: Array<Dimobj> = [];
                        const paraYdata: number[][] = [];
                        // experiment interface search space obj
                        const searchRange = JSON.parse(res1.data.params.searchSpace);
                        const reallySearchKeys = Object.keys(searchRange);
                        // trial-jobs interface list
                        Object.keys(accParaData).map(item => {
                            if (accParaData[item].status === 'SUCCEEDED') {
                                const finalData = accParaData[item].finalMetricData;
                                if (finalData && accParaData[item].hyperParameters) {
                                    const result = getFinalResult(finalData);
                                    accPara.push(result);
                                    // get dim and every line specific number
                                    const temp = JSON.parse(accParaData[item].hyperParameters).parameters;
                                    eachTrialParams.push(temp);
                                }
                            }
                        });
                        const dimName = reallySearchKeys;
                        this.getParallelAxis(dimName, searchRange, parallelAxis, accPara, eachTrialParams, paraYdata);

                        // add acc
                        Object.keys(paraYdata).map(item => {
                            paraYdata[item].push(accPara[item]);
                        });

                        // according acc to sort ydata
                        if (paraYdata.length !== 0) {
                            const len = paraYdata[0].length - 1;
                            paraYdata.sort((a, b) => b[len] - a[len]);
                        }
                        if (this._isMounted) {
                            this.setState(() => ({
                                paraBack: {
                                    parallelAxis: parallelAxis,
                                    data: paraYdata
                                }
                            }));
                        }
                        const { percent, swapAxisArr, paraBack } = this.state;
                        // need to cut down the data
                        if (percent !== 0) {
                            const linesNum = paraBack.data.length;
                            const len = Math.floor(linesNum * percent);
                            paraBack.data.length = len;
                        }
                        // need to swap the yAxis
                        if (swapAxisArr.length >= 2) {
                            this.swapGraph(paraBack, swapAxisArr);
                        }
                        this.getOption(paraBack);
                    }
                }
            }));
    }

    // get percent value number
    percentNum = (value: string) => {

        window.clearInterval(Para.intervalIDPara);
        let vals = parseFloat(value);
        if (this._isMounted) {
            this.setState(() => ({
                percent: vals
            }));
        }
        this.hyperParaPic();
        Para.intervalIDPara = window.setInterval(this.hyperParaPic, 10000);
    }

    // deal with response data into pic data
    getOption = (dataObj: ParaObj) => {
        const { visualValue } = this.state;
        let parallelAxis = dataObj.parallelAxis;
        let paralleData = dataObj.data;
        const maxAccuracy = visualValue.maxAccuracy;
        const minAccuracy = visualValue.minAccuracy;
        let visualMapObj = {};
        if (maxAccuracy === minAccuracy) {
            visualMapObj = {
                type: 'continuous',
                precision: 3,
                color: ['#CA0000', '#FFC400', '#90EE90']
            };
        } else {
            visualMapObj = {
                bottom: '20px',
                type: 'continuous',
                precision: 3,
                min: visualValue.minAccuracy,
                max: visualValue.maxAccuracy,
                color: ['#CA0000', '#FFC400', '#90EE90'],
                calculable: true
            };
        }
        let optionown = {
            parallelAxis,
            tooltip: {
                trigger: 'item'
            },
            parallel: {
                parallelAxisDefault: {
                    tooltip: {
                        show: true
                    },
                    axisLabel: {
                        formatter: function (value: string) {
                            const length = value.length;
                            if (length > 16) {
                                const temp = value.split('');
                                for (let i = 16; i < temp.length; i += 17) {
                                    temp[i] += '\n';
                                }
                                return temp.join('');
                            } else {
                                return value;
                            }
                        }
                    },
                }
            },
            visualMap: visualMapObj,
            series: {
                type: 'parallel',
                smooth: true,
                lineStyle: {
                    width: 2
                },
                data: paralleData
            }
        };
        // please wait the data
        if (this._isMounted) {
            if (paralleData.length === 0) {
                this.setState({
                    paraNodata: 'No data'
                });
            } else {
                this.setState({
                    paraNodata: ''
                });
            }
        }
        // draw search space graph
        if (this._isMounted) {
            this.setState(() => ({
                option: optionown
            }));
        }
    }

    // get swap parallel axis
    getSwapArr = (value: Array<string>) => {

        if (this._isMounted) {
            this.setState(() => ({ swapAxisArr: value }));
        }
    }

    swapBtn = () => {

        window.clearInterval(Para.intervalIDPara);
        this.hyperParaPic();
        Para.intervalIDPara = window.setInterval(this.hyperParaPic, 10000);
    }

    sortDimY = (a: Dimobj, b: Dimobj) => {
        return a.dim - b.dim;
    }

    // deal with after swap data into pic
    swapGraph = (paraBack: ParaObj, swapAxisArr: string[]) => {

        if (swapAxisArr.length >= 2) {
            const paralDim = paraBack.parallelAxis;
            const paraData = paraBack.data;
            let temp: number;
            let dim1: number;
            let dim2: number;
            let bool1: boolean = false;
            let bool2: boolean = false;
            let bool3: boolean = false;
            Object.keys(paralDim).map(item => {
                const paral = paralDim[item];
                switch (paral.name) {
                    case swapAxisArr[0]:
                        dim1 = paral.dim;
                        bool1 = true;
                        break;

                    case swapAxisArr[1]:
                        dim2 = paral.dim;
                        bool2 = true;
                        break;

                    default:
                }
                if (bool1 && bool2) {
                    bool3 = true;
                }
            });
            // swap dim's number
            Object.keys(paralDim).map(item => {
                if (bool3) {
                    if (paralDim[item].name === this.state.swapAxisArr[0]) {
                        paralDim[item].dim = dim2;
                    }
                    if (paralDim[item].name === this.state.swapAxisArr[1]) {
                        paralDim[item].dim = dim1;
                    }
                }
            });
            paralDim.sort(this.sortDimY);
            // swap data array
            Object.keys(paraData).map(paraItem => {

                temp = paraData[paraItem][dim1];
                paraData[paraItem][dim1] = paraData[paraItem][dim2];
                paraData[paraItem][dim2] = temp;
            });
        }
    }

    componentDidMount() {

        this._isMounted = true;
        // default draw all data pic
        this.hyperParaPic();
        Para.intervalIDPara = window.setInterval(this.hyperParaPic, 10000);
    }

    componentWillUnmount() {

        this._isMounted = false;
        window.clearInterval(Para.intervalIDPara);
    }

    render() {
        const { option, paraNodata, dimName } = this.state;
        const chartMulineStyle = {
            width: '100%',
            height: 392,
            margin: '0 auto',
            padding: '0 15 10 15'
        };
        return (
            <Row className="parameter">
                <Row>
                    <Col span={6} />
                    <Col span={18}>
                        <Row className="meline">
                            <span>top</span>
                            <Select
                                style={{ width: '20%', marginRight: 10 }}
                                placeholder="100%"
                                optionFilterProp="children"
                                onSelect={this.percentNum}
                            >
                                <Option value="0.2">20%</Option>
                                <Option value="0.5">50%</Option>
                                <Option value="0.8">80%</Option>
                                <Option value="1">100%</Option>
                            </Select>
                            <Select
                                style={{ width: '60%' }}
                                mode="multiple"
                                placeholder="Please select two items to swap"
                                onChange={this.getSwapArr}
                                maxTagCount={2}
                            >
                                {
                                    dimName.map((key, item) => {
                                        return (
                                            <Option key={key} value={dimName[item]}>{dimName[item]}</Option>
                                        );
                                    })
                                }
                            </Select>
                            <Button
                                type="primary"
                                className="changeBtu tableButton"
                                onClick={this.swapBtn}
                            >
                                Confirm
                            </Button>
                        </Row>
                    </Col>
                </Row>
                <Row className="searcHyper">
                    <ReactEcharts
                        option={option}
                        style={chartMulineStyle}
                        // lazyUpdate={true}
                        notMerge={true} // update now
                    />
                    <div className="noneData">{paraNodata}</div>
                </Row>
            </Row>
        );
    }
}

export default Para;