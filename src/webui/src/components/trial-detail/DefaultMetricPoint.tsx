import * as React from 'react';
import ReactEcharts from 'echarts-for-react';
import { TableObj, DetailAccurPoint, TooltipForAccuracy } from '../../static/interface';
require('echarts/lib/chart/scatter');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');

interface DefaultPointProps {
    showSource: Array<TableObj>;
    height: number;
}

interface DefaultPointState {
    defaultSource: object;
    accNodata: string;
}

class DefaultPoint extends React.Component<DefaultPointProps, DefaultPointState> {
    public _isMounted = false;

    constructor(props: DefaultPointProps) {
        super(props);
        this.state = {
            defaultSource: {},
            accNodata: 'No data'
        };
    }

    defaultMetric = (showSource: Array<TableObj>) => {
        const accSource: Array<DetailAccurPoint> = [];
        Object.keys(showSource).map(item => {
            const temp = showSource[item];
            if (temp.status === 'SUCCEEDED' && temp.acc.default !== undefined) {
                const searchSpace = temp.description.parameters;
                accSource.push({
                    acc: temp.acc.default,
                    index: temp.sequenceId,
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
                        '<div>Trial No: ' + data.data[0] + '</div>' +
                        '<div>Default Metric: ' + data.data[1] + '</div>' +
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
        if (this._isMounted === true) {
            this.setState({ defaultSource: allAcuracy }, () => {
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
    }

    // update parent component state
    componentWillReceiveProps(nextProps: DefaultPointProps) {
        const showSource = nextProps.showSource;
        this.defaultMetric(showSource);
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
                />
                <div className="showMess">{accNodata}</div>
            </div>
        );
    }
}

export default DefaultPoint;