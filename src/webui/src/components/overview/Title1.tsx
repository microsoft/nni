import * as React from 'react';
import { Stack } from '@fluentui/react';
import {tableListIcon} from '../buttons/Icon';
import { TitleContext } from '../Overview';
import '../../static/style/overview/overviewTitle.scss';

export const Title1 = (): any => (
    <TitleContext.Consumer>
        {(value): React.ReactNode => (
            <Stack horizontal className='panelTitle'>
                <img src={require(`../../static/img/icon/${value.icon}`)} alt='icon' />
                <span style={{ color: value.fontColor }}>{value.text}</span>
            </Stack>
        )}
    </TitleContext.Consumer>
);

/**
 * Experiment: TestParameter | AutoRacing
 * Duration: Timer
 * Trial numbers: NumberSymbol
 * Trial metric chart: HomeGroup
 * Top trials: BulletedList
 * succeed table button: MarketDown Market
 * 
 * 
 * 
 * Trial detail
 * defatult metric: HomeGroup 
 * hyper-parameter:
 * duration: BarChartHorizontal
 * Intermediate: StackedLineChart
 * table: BulletedList
 */