import * as React from 'react';
import ReactEcharts from 'echarts-for-react';
import { filterByStatus } from '../../static/function';
import { EXPERIMENT } from '../../static/datamodel';
import { Stack, PrimaryButton, Dropdown, IDropdownOption, } from 'office-ui-fabric-react'; // eslint-disable-line no-unused-vars
import { ParaObj, Dimobj, TableObj } from '../../static/interface'; // eslint-disable-line no-unused-vars
import 'echarts/lib/chart/parallel';
import 'echarts/lib/component/tooltip';
import 'echarts/lib/component/title';
import 'echarts/lib/component/visualMap';
import '../../static/style/para.scss';
import '../../static/style/button.scss';

interface ParaState {
    // paraSource: Array<TableObj>;
    option: object;
    paraBack: ParaObj;
    dimName: string[];
    swapAxisArr: string[];
    percent: number;
    paraNodata: string;
    max: number; // graph color bar limit
    min: number;
    sutrialCount: number; // succeed trial numbers for SUC
    succeedRenderCount: number; // all succeed trials number
    clickCounts: number;
    isLoadConfirm: boolean;
    // office-fabric-ui
    selectedItem?: { key: string | number | undefined }; // percent Selector
    swapyAxis?: string[]; // yAxis Selector
}

interface ParaProps {
    dataSource: Array<TableObj>;
    expSearchSpace: string;
    whichGraph: string;
}

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
            isLoadConfirm: false,
            swapyAxis: []
        };
    }

    getParallelAxis =
        (
            dimName: string[], parallelAxis: Array<Dimobj>,
            accPara: number[], eachTrialParams: string[],
            lengthofTrials: number
        ): void => {
            // get data for every lines. if dim is choice type, number -> toString()
            const paraYdata: number[][] = [];
            Object.keys(eachTrialParams).map(item => {
                const temp: number[] = [];
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
                // show top trials
                if (EXPERIMENT.optimizeMode === 'minimize') {
                    paraYdata.sort((a, b) => a[len] - b[len]);
                }
                if (EXPERIMENT.optimizeMode === 'maximize') {
                    paraYdata.sort((a, b) => b[len] - a[len]);
                }
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

    hyperParaPic = (source: Array<TableObj>, searchSpace: string): void => {
        // filter succeed trials [{}, {}, {}]
        const dataSource = source.filter(filterByStatus);
        const lenOfDataSource: number = dataSource.length;
        const accPara: number[] = [];
        // specific value array
        const eachTrialParams: string[] = [];
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
                const data: string[] = [];
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
                const data: string[] = [];
                let j = 0;
                switch (searchKey._type) {
                    case 'choice':
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
                            formatter: function (value?: string): string | null {
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
                        const item = element[key];
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
    // percentNum = (value: string) => {
    percentNum = (event: React.FormEvent<HTMLDivElement>, item?: IDropdownOption): void => {
        // percentNum = (event: React.FormEvent<HTMLDivElement>, item?: ISelectableOption) => {
        if (item !== undefined) {
            const vals = parseFloat(item !== undefined ? item.text : '');
            this.setState({ percent: vals / 100, selectedItem: item }, () => {
                this.reInit();
            });
        }
    }

    // deal with response data into pic data
    getOption = (dataObj: ParaObj, lengthofTrials: number): void => {
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
                        formatter: function (value: string): string {
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
    getSwapArr = (event: React.FormEvent<HTMLDivElement>, item?: IDropdownOption): void => {
        const newSelectedItems = [...this.state.swapyAxis];
        if (item !== undefined) {
            if (item.selected) {
                // add the option if it's checked
                newSelectedItems.push(item.key as string);
            } else {
                // remove the option if it's unchecked
                const currIndex = newSelectedItems.indexOf(item.key as string);
                if (currIndex > -1) {
                    newSelectedItems.splice(currIndex, 1);
                }
            }
            this.setState({
                swapAxisArr: newSelectedItems,
                swapyAxis: newSelectedItems
            });
        }
    }

    reInit = (): void => {
        const { dataSource, expSearchSpace } = this.props;
        this.hyperParaPic(dataSource, expSearchSpace);
    }

    swapReInit = (): void => {
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

    sortDimY = (a: Dimobj, b: Dimobj): number => {
        return a.dim - b.dim;
    }

    // deal with after swap data into pic
    swapGraph = (paraBack: ParaObj, swapAxisArr: string[]): void => {
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

    componentDidMount(): void {
        this.reInit();
    }

    componentDidUpdate(prevProps: ParaProps): void {
        if(this.props.dataSource !== prevProps.dataSource) {
            const { dataSource, expSearchSpace, whichGraph } = this.props;
            if (whichGraph === 'Hyper-parameter') {
                this.hyperParaPic(dataSource, expSearchSpace);
            }
        }
    }

    render(): React.ReactNode {
        const { option, paraNodata, dimName, isLoadConfirm, selectedItem, swapyAxis } = this.state;
        return (
            <div className="parameter">
                <Stack horizontal className="para-filter" horizontalAlign="end">
                    <span className="para-filter-text">Top</span>
                    <Dropdown
                        selectedKey={selectedItem ? selectedItem.key : undefined}
                        onChange={this.percentNum}
                        placeholder="100%"
                        defaultSelectedKeys={[0.2]}
                        options={[
                            { key: '0.2', text: '20%' },
                            { key: '0.5', text: '50%' },
                            { key: '0.8', text: '80%' },
                            { key: '1', text: '100%' },
                        ]}
                        styles={{ dropdown: { width: 300 } }}
                        className="para-filter-percent"
                    />
                    <Dropdown
                        placeholder="Select options"
                        selectedKeys={swapyAxis}
                        onChange={this.getSwapArr}
                        multiSelect
                        options={
                            dimName.map((key, item) => {
                                return {
                                    key: key, text: dimName[item]
                                };
                            })
                        }
                        styles={{ dropdown: { width: 300 } }}
                    />
                    <PrimaryButton
                        text="Confirm"
                        onClick={this.swapReInit}
                        disabled={isLoadConfirm}
                    />
                </Stack>
                <div className="searcHyper">
                    <ReactEcharts
                        option={option}
                        style={this.chartMulineStyle}
                        // lazyUpdate={true}
                        notMerge={true} // update now
                    />
                    <div className="noneData">{paraNodata}</div>
                </div>
            </div>
        );
    }
}

export default Para;
