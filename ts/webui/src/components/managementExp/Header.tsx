import React from 'react';
import { Link } from 'react-router-dom';
import { Stack, StackItem, CommandBarButton } from '@fluentui/react';
import { RevToggleKey } from '../buttons/Icon';
import { NNILOGO } from '../stateless-component/NNItabs';
import { stackTokens, stackStyle } from '../NavConst';

export const Hearder = (): any => (
    <div className='header'>
        <div className='headerCon'>
            <Stack className='nav' horizontal>
                <StackItem grow={30} styles={{ root: { minWidth: 300, display: 'flex', verticalAlign: 'center' } }}>
                    <span className='desktop-logo'>{NNILOGO}</span>
                    <span className='logoTitle'>Neural Network Intelligence</span>
                </StackItem>
                <StackItem grow={70} className='navOptions'>
                    <Stack horizontal horizontalAlign='end' tokens={stackTokens} styles={stackStyle}>
                        <Link to='/oview' className='experiment'>
                            <CommandBarButton iconProps={RevToggleKey} text='Back to the experiment' />
                        </Link>
                    </Stack>
                </StackItem>
            </Stack>
        </div>
    </div>
);
