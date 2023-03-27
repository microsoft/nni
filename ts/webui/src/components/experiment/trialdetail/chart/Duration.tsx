import React from 'react';
import ReactEcharts from 'echarts-for-react';
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

const drawDurationOptions = (dataObj: Runtrial): any => {
    return {
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            },
            enterable: true,
            formatter: (data: any): React.ReactNode =>
                `<div class="tooldetailAccuracy">
                    <div class='main'>
                        <div>Trial No.: ${data[0].dataIndex}</div>
                        <div>Duration: ${convertDuration(data[0].data)}</div>
                    </div>
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
                start: 0,
                end: 100 // percent
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

const Duration = React.memo((props: DurationProps) => {
    const { source } = props;
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
    const options = drawDurationOptions(trialRun[0]);

    return (
        <div className='graph'>
            <ReactEcharts
                option={options}
                style={{ width: '94%', height: 412, margin: '0 auto', marginTop: 15 }}
                theme='nni_theme'
                // notMerge={true} // update now
                lazyUpdate={true}
            />
        </div>
    );
});

export default Duration;
