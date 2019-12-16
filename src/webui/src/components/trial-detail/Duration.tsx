import * as React from 'react';
import ReactEcharts from 'echarts-for-react';
import { TableObj, EventMap } from 'src/static/interface';
import { filterDuration } from 'src/static/function';
require('echarts/lib/chart/bar');
require('echarts/lib/component/tooltip');
require('echarts/lib/component/title');

interface Runtrial {
    trialId: Array<string>;
    trialTime: Array<number>;
}

interface DurationProps {
    source: Array<TableObj>;
    whichGraph: string;
}

interface DurationState {
    startDuration: number; // for record data zoom
    endDuration: number;
}

class Duration extends React.Component<DurationProps, DurationState> {

    constructor(props: DurationProps) {

        super(props);
        this.state = {
            startDuration: 0, // for record data zoom
            endDuration: 100,
        };
    }

    getOption = (dataObj: Runtrial) => {
        const { startDuration, endDuration } = this.state;
        return {
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'shadow'
                }
            },
            grid: {
                bottom: '3%',
                containLabel: true,
                left: '1%',
                right: '4%'
            },
            dataZoom: [
                {
                    id: 'dataZoomY',
                    type: 'inside',
                    yAxisIndex: [0],
                    filterMode: 'empty',
                    start: startDuration,
                    end: endDuration
                },
            ],
            xAxis: {
                name: 'Time',
                type: 'value',
            },
            yAxis: {
                name: 'Trial',
                type: 'category',
                data: dataObj.trialId
            },
            series: [{
                type: 'bar',
                data: dataObj.trialTime
            }]
        };
    }

    drawDurationGraph = (source: Array<TableObj>) => {
        // why this function run two times when props changed?
        const trialId: Array<string> = [];
        const trialTime: Array<number> = [];
        const trialRun: Array<Runtrial> = [];
        const trialJobs = source.filter(filterDuration);
        Object.keys(trialJobs).map(item => {
            const temp = trialJobs[item];
            trialId.push(temp.sequenceId);
            trialTime.push(temp.duration);
        });
        trialRun.push({
            trialId: trialId,
            trialTime: trialTime
        });
        return this.getOption(trialRun[0]);
    }

    shouldComponentUpdate(nextProps: DurationProps, nextState: DurationState) {

        const { whichGraph, source } = nextProps;
        if (whichGraph === '3') {
            const beforeSource = this.props.source;
            if (whichGraph !== this.props.whichGraph) {
                return true;
            }

            if (source.length !== beforeSource.length) {
                return true;
            }

            if (beforeSource[beforeSource.length - 1] !== undefined) {
                if (source[source.length - 1].duration !== beforeSource[beforeSource.length - 1].duration) {
                    return true;
                }
                if (source[source.length - 1].status !== beforeSource[beforeSource.length - 1].status) {
                    return true;
                }
            }
        }
        return false;
    }

    render() {

        const { source } = this.props;
        const graph = this.drawDurationGraph(source);
        const onEvents = { 'dataZoom': this.durationDataZoom };
        return (
            <div>
                <ReactEcharts
                    option={graph}
                    style={{ width: '95%', height: 412, margin: '0 auto' }}
                    theme="my_theme"
                    notMerge={true} // update now
                    onEvents={onEvents}
                />
            </div>
        );
    }

    private durationDataZoom = (e: EventMap) => {
        if (e.batch !== undefined) {
            this.setState(() => ({
                startDuration: (e.batch[0].start !== null ? e.batch[0].start : 0),
                endDuration: (e.batch[0].end !== null ? e.batch[0].end : 100)
            }));
        }
    }
}

export default Duration;
