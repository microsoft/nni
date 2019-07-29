import * as React from 'react';
import ReactEcharts from 'echarts-for-react';
import { filterByStatus } from '../../static/function';
import { TableObj, DetailAccurPoint, TooltipForAccuracy } from '../../static/interface';
require('echarts/lib/chart/scatter');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');

interface DefaultPointProps {
    showSource: Array<TableObj>;
    height: number;
    whichGraph: string;
}

interface DefaultPointState {
    defaultSource: object;
    accNodata: string;
    succeedTrials: number;
}

class DefaultPoint extends React.Component<DefaultPointProps, DefaultPointState> {
    public _isMounted = false;

    constructor(props: DefaultPointProps) {
        super(props);
        this.state = {
            defaultSource: {},
            accNodata: '',
            succeedTrials: 10000000
        };
    }

    defaultMetric = (succeedSource: Array<TableObj>) => {
        const accSource: Array<DetailAccurPoint> = [];
        const showSource: Array<TableObj> = succeedSource.filter(filterByStatus);
        const lengthOfSource = showSource.length;
        const tooltipDefault = lengthOfSource === 0 ? 'No data' : '';
        if (this._isMounted === true) {
            this.setState(() => ({
                succeedTrials: lengthOfSource,
                accNodata: tooltipDefault
            }));
        }
        if (lengthOfSource === 0) {
            const nullGraph = {
                grid: {
                    left: '8%'
                },
                xAxis: {
                    name: 'Trial',
                    type: 'category',
                },
                yAxis: {
                    name: 'Default metric',
                    type: 'value',
                }
            };
            if (this._isMounted === true) {
                this.setState(() => ({
                    defaultSource: nullGraph
                }));
            }
        } else {
            const resultList: Array<number | string>[] = [];
            const lineListDefault: Array<number | string> = [];
            Object.keys(showSource).map(item => {
                const temp = showSource[item];
                if (temp.acc !== undefined) {
                    if (temp.acc.default !== undefined) {
                        const searchSpace = temp.description.parameters;
                        lineListDefault.push(temp.acc.default);
                        accSource.push({
                            acc: temp.acc.default,
                            index: temp.sequenceId,
                            searchSpace: JSON.stringify(searchSpace)
                        });
                    }
                }
            });
            // deal with best metric line
            const realDefault: Array<number | string> = []; // 真正的line图数据
            realDefault.push(lineListDefault[0]); // 放进来第一个值
            let realNum = lineListDefault[0]; // 当前数组最后一个number
            for (let i = 1; i < lineListDefault.length; i++) {
                const val = lineListDefault[i];
                if (realDefault[realDefault.length - 1] === '') {
                    if (val >= realNum) {
                        realDefault.push(val);
                        realNum = val;
                    } else {
                        realDefault.push('');
                    }
                } else {
                    if (val >= realDefault[realDefault.length - 1]) {
                        realDefault.push(val);
                        realNum = val;
                    } else {
                        realDefault.push('');
                    }
                }
            }
            console.info('want', realDefault);
            Object.keys(accSource).map(item => {
                const items = accSource[item];
                let temp: Array<number | string>;
                temp = [items.index, items.acc, JSON.parse(items.searchSpace)];
                resultList.push(temp);
            });

            const allAcuracy = {
                grid: {
                    left: '8%'
                },
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
                            '<div>Trial No.: ' + data.data[0] + '</div>' +
                            '<div>Default metric: ' + data.data[1] + '</div>' +
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
                    name: 'Default metric',
                    type: 'value',
                },
                series: [{
                    symbolSize: 6,
                    type: 'scatter',
                    data: resultList
                },
                // add line graph
                {
                    type: 'line',
                    lineStyle: {
                        color: 'red'
                    },
                    connectNulls: true, // avoid break point in line
                    data: realDefault
                }]
            };
            if (this._isMounted === true) {
                this.setState(() => ({
                    defaultSource: allAcuracy
                }));
            }
        }
    }

    // update parent component state
    componentWillReceiveProps(nextProps: DefaultPointProps) {

        const { whichGraph, showSource } = nextProps;
        if (whichGraph === '1') {
            this.defaultMetric(showSource);
        }
    }

    shouldComponentUpdate(nextProps: DefaultPointProps, nextState: DefaultPointState) {
        const { whichGraph } = nextProps;
        const succTrial = this.state.succeedTrials;
        const { succeedTrials } = nextState;
        if (whichGraph === '1') {
            if (succeedTrials !== succTrial) {
                return true;
            }
        }
        // only whichGraph !== '1', default metric can't update
        return false;
    }

    componentDidMount() {
        this._isMounted = true;
    }

    componentWillUnmount() {
        this._isMounted = false;
    }

    render() {
        const { height } = this.props;
        const { defaultSource, accNodata } = this.state;
        return (
            <div>
                <ReactEcharts
                    option={defaultSource}
                    style={{
                        width: '100%',
                        height: height,
                        margin: '0 auto',
                    }}
                    theme="my_theme"
                    notMerge={true} // update now
                // lazyUpdate={true}
                />
                <div className="showMess">{accNodata}</div>
            </div>
        );
    }
}

export default DefaultPoint;