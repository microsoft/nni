import * as React from 'react';
import ReactEcharts from 'echarts-for-react';
import { filterByStatus } from '../../static/function';
import { Row, Col, Select, Button, message } from 'antd';
import { ParaObj, Dimobj, TableObj } from '../../static/interface';
const Option = Select.Option;
require('echarts/lib/chart/parallel');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');
require('echarts/lib/component/visualMap');
require('../../static/style/para.scss');
require('../../static/style/button.scss');

interface ParaState {
    // paraSource: Array<TableObj>;
    option: object;
    paraBack: ParaObj;
    dimName: Array<string>;
    swapAxisArr: Array<string>;
    percent: number;
    paraNodata: string;
    max: number; // graph color bar limit
    min: number;
    sutrialCount: number; // succeed trial numbers for SUC
    succeedRenderCount: number; // all succeed trials number
    clickCounts: number;
    isLoadConfirm: boolean;
}

interface ParaProps {
    dataSource: Array<TableObj>;
    expSearchSpace: string;
    whichGraph: string;
}

message.config({
    top: 250,
    duration: 2,
});

class Para extends React.Component<ParaProps, ParaState> {

    private chartMulineStyle = {
        width: '100%',
        height: 392,
        margin: '0 auto',
        padding: '0 15 10 15'
    };

    constructor(props: ParaProps) {
        super(props);
        this.state = {
            // paraSource: [],
            // option: this.hyperParaPic,
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
            min: 0,
            max: 1,
            sutrialCount: 10000000,
            succeedRenderCount: 10000000,
            clickCounts: 1,
            isLoadConfirm: false
        };
    }

    getParallelAxis =
        (
            dimName: Array<string>, parallelAxis: Array<Dimobj>,
            accPara: Array<number>, eachTrialParams: Array<string>,
            lengthofTrials: number
        ) => {
            // get data for every lines. if dim is choice type, number -> toString()
            const paraYdata: number[][] = [];
            Object.keys(eachTrialParams).map(item => {
                let temp: Array<number> = [];
                for (let i = 0; i < dimName.length; i++) {
                    if ('type' in parallelAxis[i]) {
                        temp.push(eachTrialParams[item][dimName[i]].toString());
                    } else {
                        // default metric
                        temp.push(eachTrialParams[item][dimName[i]]);
                    }
                }
                paraYdata.push(temp);
            });
            // add acc
            Object.keys(paraYdata).map(item => {
                paraYdata[item].push(accPara[item]);
            });
            // according acc to sort ydata // sort to find top percent dataset
            if (paraYdata.length !== 0) {
                const len = paraYdata[0].length - 1;
                paraYdata.sort((a, b) => b[len] - a[len]);
            }
            const paraData = {
                parallelAxis: parallelAxis,
                data: paraYdata
            };
            const { percent, swapAxisArr } = this.state;
            // need to cut down the data
            if (percent !== 0) {
                const linesNum = paraData.data.length;
                // Math.ceil rather than Math.floor to avoid lost lines
                const len = Math.ceil(linesNum * percent);
                paraData.data.length = len;
            }
            // need to swap the yAxis
            if (swapAxisArr.length >= 2) {
                this.swapGraph(paraData, swapAxisArr);
            }
            this.getOption(paraData, lengthofTrials);
            this.setState({ paraBack: paraData });
        }

    hyperParaPic = (source: Array<TableObj>, searchSpace: string) => {
        // filter succeed trials [{}, {}, {}]
        const dataSource = source.filter(filterByStatus);
        const lenOfDataSource: number = dataSource.length;
        const accPara: Array<number> = [];
        // specific value array
        const eachTrialParams: Array<string> = [];
        // experiment interface search space obj
        const searchRange = searchSpace !== undefined ? JSON.parse(searchSpace) : '';
        // nest search space
        let isNested: boolean = false;
        Object.keys(searchRange).map(item => {
            if (searchRange[item]._value && typeof searchRange[item]._value[0] === 'object') {
                isNested = true;
                return;
            }
        });
        const dimName = Object.keys(searchRange);
        this.setState({ dimName: dimName });

        const parallelAxis: Array<Dimobj> = [];
        // search space range and specific value [only number]
        let i = 0;
        if (isNested === false) {
            for (i; i < dimName.length; i++) {
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
                            min: searchKey._value[0],
                            max: searchKey._value[1],
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
                            data: data,
                            boundaryGap: true,
                            axisLine: {
                                lineStyle: {
                                    type: 'dotted', // axis type,solid，dashed，dotted
                                    width: 1
                                }
                            },
                            axisTick: {
                                show: true,
                                interval: 0,
                                alignWithLabel: true,
                            },
                            axisLabel: {
                                show: true,
                                interval: 0,
                                // rotate: 30
                            },
                        });
                        break;
                    // support log distribute
                    case 'loguniform':
                        if (lenOfDataSource > 1) {
                            parallelAxis.push({
                                dim: i,
                                name: dimName[i],
                                type: 'log',
                            });
                        } else {
                            parallelAxis.push({
                                dim: i,
                                name: dimName[i]
                            });
                        }
                        break;
                    default:
                        parallelAxis.push({
                            dim: i,
                            name: dimName[i]
                        });
                }
            }
        } else {
            for (i; i < dimName.length; i++) {
                const searchKey = searchRange[dimName[i]];
                switch (searchKey._type) {
                    case 'choice':
                        const data: Array<string> = [];
                        let j = 0;
                        for (j; j < searchKey._value.length; j++) {
                            const item = searchKey._value[j];
                            Object.keys(item).map(key => {
                                if (key !== '_name' && key !== '_type') {
                                    Object.keys(item[key]).map(index => {
                                        if (index !== '_type') {
                                            const realChoice = item[key][index];
                                            Object.keys(realChoice).map(m => {
                                                data.push(`${item._name}_${realChoice[m]}`);
                                            });
                                        }
                                    });
                                }
                            });
                        }
                        data.push('null');
                        parallelAxis.push({
                            dim: i,
                            name: dimName[i],
                            type: 'category',
                            data: data,
                            boundaryGap: true,
                            axisLine: {
                                lineStyle: {
                                    type: 'dotted', // axis type,solid dashed dotted
                                    width: 1
                                }
                            },
                            axisTick: {
                                show: true,
                                interval: 0,
                                alignWithLabel: true,
                            },
                            axisLabel: {
                                show: true,
                                interval: 0,
                                // rotate: 30
                            },
                        });
                        break;
                    default:
                        parallelAxis.push({
                            dim: i,
                            name: dimName[i]
                        });
                }
            }
        }
        parallelAxis.push({
            dim: i,
            name: 'default metric',
            scale: true,
            nameTextStyle: {
                fontWeight: 700
            }
        });
        if (lenOfDataSource === 0) {
            const optionOfNull = {
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
                            formatter: function (value?: string) {
                                if (value !== undefined) {
                                    const length = value.length;
                                    if (length > 16) {
                                        const temp = value.split('');
                                        for (let m = 16; m < temp.length; m += 17) {
                                            temp[m] += '\n';
                                        }
                                        return temp.join('');
                                    } else {
                                        return value;
                                    }
                                } else {
                                    return null;
                                }
                            }
                        },
                    }
                },
                visualMap: {
                    type: 'continuous',
                    min: 0,
                    max: 1,
                    color: ['#CA0000', '#FFC400', '#90EE90']
                }
            };
            this.setState({
                paraNodata: 'No data',
                option: optionOfNull,
                sutrialCount: 0,
                succeedRenderCount: 0
            });
        } else {
            Object.keys(dataSource).map(item => {
                const trial = dataSource[item];
                eachTrialParams.push(trial.description.parameters || '');
                // may be a succeed trial hasn't final result
                // all detail page may be break down if havn't if
                if (trial.acc !== undefined) {
                    if (trial.acc.default !== undefined) {
                        accPara.push(JSON.parse(trial.acc.default));
                    }
                }
            });
            // nested search space, deal data
            if (isNested !== false) {
                eachTrialParams.forEach(element => {
                    Object.keys(element).forEach(key => {
                        let item = element[key];
                        if (typeof item === 'object') {
                            Object.keys(item).forEach(index => {
                                if (index !== '_name') {
                                    element[key] = `${item._name}_${item[index]}`;
                                } else {
                                    element[key] = 'null';
                                }
                            });
                        }
                    });
                });
            }
            // if not return final result
            const maxVal = accPara.length === 0 ? 1 : Math.max(...accPara);
            const minVal = accPara.length === 0 ? 1 : Math.min(...accPara);
            this.setState({ max: maxVal, min: minVal }, () => {
                this.getParallelAxis(dimName, parallelAxis, accPara, eachTrialParams, lenOfDataSource);
            });
        }
    }

    // get percent value number
    percentNum = (value: string) => {

        let vals = parseFloat(value);
        this.setState({ percent: vals }, () => {
            this.reInit();
        });
    }

    // deal with response data into pic data
    getOption = (dataObj: ParaObj, lengthofTrials: number) => {
        // dataObj [[y1], [y2]... [default metric]]
        const { max, min } = this.state;
        const parallelAxis = dataObj.parallelAxis;
        const paralleData = dataObj.data;
        let visualMapObj = {};
        if (max === min) {
            visualMapObj = {
                type: 'continuous',
                precision: 3,
                min: 0,
                max: max,
                color: ['#CA0000', '#FFC400', '#90EE90']
            };
        } else {
            visualMapObj = {
                bottom: '20px',
                type: 'continuous',
                precision: 3,
                min: min,
                max: max,
                color: ['#CA0000', '#FFC400', '#90EE90']
            };
        }
        const optionown = {
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
        this.setState({
            option: optionown,
            paraNodata: '',
            succeedRenderCount: lengthofTrials,
            sutrialCount: paralleData.length
        });
    }

    // get swap parallel axis
    getSwapArr = (value: Array<string>) => {
        this.setState({ swapAxisArr: value });
    }

    reInit = () => {
        const { dataSource, expSearchSpace } = this.props;
        this.hyperParaPic(dataSource, expSearchSpace);
    }

    swapReInit = () => {
        const { clickCounts, succeedRenderCount } = this.state;
        const val = clickCounts + 1;
        this.setState({ isLoadConfirm: true, clickCounts: val, });
        const { paraBack, swapAxisArr } = this.state;
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
                if (paralDim[item].name === swapAxisArr[0]) {
                    paralDim[item].dim = dim2;
                }
                if (paralDim[item].name === swapAxisArr[1]) {
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
        this.getOption(paraBack, succeedRenderCount);
        // please wait the data
        this.setState({
            isLoadConfirm: false
        });
    }

    sortDimY = (a: Dimobj, b: Dimobj) => {
        return a.dim - b.dim;
    }

    // deal with after swap data into pic
    swapGraph = (paraBack: ParaObj, swapAxisArr: string[]) => {
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
                if (paralDim[item].name === swapAxisArr[0]) {
                    paralDim[item].dim = dim2;
                }
                if (paralDim[item].name === swapAxisArr[1]) {
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

    componentDidMount() {
        this.reInit();
    }

    componentWillReceiveProps(nextProps: ParaProps) {
        const { dataSource, expSearchSpace, whichGraph } = nextProps;
        if (whichGraph === '2') {
            this.hyperParaPic(dataSource, expSearchSpace);
        }
    }

    shouldComponentUpdate(nextProps: ParaProps, nextState: ParaState) {

        const { whichGraph } = nextProps;
        const beforeGraph = this.props.whichGraph;
        if (whichGraph === '2') {
            if (whichGraph !== beforeGraph) {
                return true;
            }

            const { sutrialCount, clickCounts, succeedRenderCount } = nextState;
            const beforeCount = this.state.sutrialCount;
            const beforeClickCount = this.state.clickCounts;
            const beforeRealRenderCount = this.state.succeedRenderCount;
            if (sutrialCount !== beforeCount) {
                return true;
            }
            if (succeedRenderCount !== beforeRealRenderCount) {
                return true;
            }

            if (clickCounts !== beforeClickCount) {
                return true;
            }
        }
        return false;
    }

    render() {
        const { option, paraNodata, dimName, isLoadConfirm } = this.state;
        return (
            <Row className="parameter">
                <Row>
                    <Col span={6} />
                    <Col span={18}>
                        <Row className="meline">
                            <span>Top</span>
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
                                onClick={this.swapReInit}
                                disabled={isLoadConfirm}
                            >
                                Confirm
                            </Button>
                        </Row>
                    </Col>
                </Row>
                <Row className="searcHyper">
                    <ReactEcharts
                        option={option}
                        style={this.chartMulineStyle}
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
