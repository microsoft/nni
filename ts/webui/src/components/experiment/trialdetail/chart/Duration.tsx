import React, { useState, useEffect } from 'react';
import ReactEcharts from 'echarts-for-react';
import { EventMap } from '@static/interface';
import { Trial } from '@model/trial';
import { convertDuration } from '@static/function';
import 'echarts/lib/chart/bar';
import 'echarts/lib/component/tooltip';
import 'echarts/lib/component/title';

interface Runtrial {
    trialId: number[];
    trialTime: number[];
}

interface DurationProps {
    source: Trial[];
}

const initDuration = (source: Trial[]): any => {
    const trialId: number[] = [];
    const trialTime: number[] = [];
    source.forEach(item => {
        trialId.push(item.sequenceId);
        trialTime.push(item.duration);
    });
    return {
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            },
            formatter: (data: any): React.ReactNode =>
                '<div>' +
                '<div>Trial No.: ' +
                data[0].dataIndex +
                '</div>' +
                '<div>Duration: ' +
                convertDuration(data[0].data) +
                '</div>' +
                '</div>'
        },
        grid: {
            bottom: '3%',
            containLabel: true,
            left: '1%',
            right: '5%'
        },
        dataZoom: [
            {
                id: 'dataZoomY',
                type: 'inside',
                yAxisIndex: [0],
                filterMode: 'empty',
                start: 0,
                end: 100
            }
        ],
        xAxis: {
            name: 'Time/s',
            type: 'value'
        },
        yAxis: {
            name: 'Trial No.',
            type: 'category',
            data: trialId,
            nameTextStyle: {
                padding: [0, 0, 0, 30]
            }
        },
        series: [
            {
                type: 'bar',
                data: trialTime
            }
        ]
    };
};

const Duration = (props: DurationProps) => {
    const { source } = props;
    const [startDuration, setStartDuration] = useState(0 as number); // for record data zoom
    const [endDuration, setEndDuration] = useState(100 as number);
    const [durationSource, setDurationSource] = useState(initDuration(source) as {}); // 数据类型泛泛
    const getOption = (dataObj: Runtrial): any => {
        return {
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'shadow'
                },
                enterable: true,
                formatter: (data: any): React.ReactNode =>
                    `<div class="tooldetailAccuracy">
                        <div>Trial No.: ${data[0].dataIndex}</div>
                        <div>Duration: ${convertDuration(data[0].data)}</div>
                    </div>
                    `
            },
            grid: {
                bottom: '3%',
                containLabel: true,
                left: '1%',
                right: '5%'
            },
            dataZoom: [
                {
                    id: 'dataZoomY',
                    type: 'inside',
                    yAxisIndex: [0],
                    filterMode: 'empty',
                    start: startDuration,
                    end: endDuration
                }
            ],
            xAxis: {
                name: 'Time',
                type: 'value'
            },
            yAxis: {
                name: 'Trial',
                type: 'category',
                data: dataObj.trialId,
                nameTextStyle: {
                    padding: [0, 0, 0, 30]
                }
            },
            series: [
                {
                    type: 'bar',
                    data: dataObj.trialTime
                }
            ]
        };
    };
    const drawDurationGraph = (source: Trial[]): void => {
        // why this function run two times when props changed?
        const trialId: number[] = [];
        const trialTime: number[] = [];
        const trialRun: Runtrial[] = [];
        source.forEach(item => {
            trialId.push(item.sequenceId);
            trialTime.push(item.duration);
        });

        trialRun.push({
            trialId: trialId,
            trialTime: trialTime
        });
        setDurationSource(getOption(trialRun[0]));
    };

    useEffect(() => {
        drawDurationGraph(source);
    }, [source]); // TODO: source 有变化时，更新页面，要测！

    const durationDataZoom = (e: EventMap): void => {
        if (e.batch !== undefined) {
            setStartDuration(e.batch[0].start !== null ? e.batch[0].start : 0);
            setEndDuration(e.batch[0].end !== null ? e.batch[0].end : 100)
        }
    };
    const onEvents = { dataZoom: durationDataZoom };

    return (
        <div className='graph'>
            <ReactEcharts
                option={durationSource}
                style={{ width: '94%', height: 412, margin: '0 auto', marginTop: 15 }}
                theme='nni_theme'
                notMerge={true} // update now
                onEvents={onEvents}
            />
        </div>
    );
};

export default Duration;
