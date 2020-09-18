import * as React from 'react';
import { Stack } from '@fluentui/react';
import { TitleContext } from '../Overview';
import '../../static/style/overviewTitle.scss';

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
