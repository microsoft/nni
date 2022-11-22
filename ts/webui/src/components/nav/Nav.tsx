import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Stack, StackItem, CommandBarButton, IContextualMenuProps } from '@fluentui/react';
import { Link } from 'react-router-dom';
import { MANAGER_IP, WEBUIDOC } from '@static/const';
import ExperimentSummaryPanel from './slideNav/ExperimentSummaryPanel';
import { OVERVIEWTABS, DETAILTABS, NNILOGO } from './slideNav/NNItabs';
import { EXPERIMENT } from '@static/datamodel';
import { gap15, stackStyle } from '@components/fluent/ChildrenGap';
import {
    infoIconAbout,
    timeIcon,
    disableUpdates,
    requency,
    closeTimer,
    ChevronRightMed
} from '@components/fluent/Icon';
import '@style/nav/nav.scss';
import '@style/icon.scss';
import { ErrorMessage } from '@components/nav/ErrorMessage';

interface NavProps {
    changeInterval: (value: number) => void;
}

const NavCon = (props: NavProps): any => {
    const { changeInterval } = props;
    const [version, setVersion] = useState('999' as string);
    const [visibleExperimentPanel, setVisibleExperimentPanel] = useState(false);
    const [refreshText, setRefreshText] = useState('Auto refresh' as string);
    const [refreshFrequency, setRefreshFrequency] = useState(10 as number | string);

    const openGithub = (): void => {
        const feed = `https://github.com/Microsoft/nni/issues/new?labels=${version}`;
        window.open(feed);
    };

    const openDocs = (): void => {
        window.open(WEBUIDOC);
    };

    const openGithubNNI = (): void => {
        // 999.0.0-developing
        let formatVersion = `v${version}`;
        if (version === '999.0.0-developing') {
            formatVersion = 'master';
        }
        const nniLink = `https://github.com/Microsoft/nni/tree/${formatVersion}`;
        window.open(nniLink);
    };

    const getInterval = (num: number): void => {
        changeInterval(num); // notice parent component
        setRefreshFrequency(num === 0 ? '' : num);
        setRefreshText(num === 0 ? 'Disable auto' : 'Auto refresh');
    };

    useEffect(() => {
        axios(`${MANAGER_IP}/version`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    let formatVersion = res.data;
                    // 2.0 will get 2.0.0 by node, so delete .0 to get real version
                    if (formatVersion.endsWith('.0')) {
                        formatVersion = formatVersion.slice(0, -2);
                    }
                    setVersion(formatVersion);
                }
            })
            .catch(_error => {
                // TODO 测试这块有没有问题，一个404的api返回status是200.。。
                setVersion('ERROR');
            });
    }, []);

    const refreshProps: IContextualMenuProps = {
        items: [
            {
                key: 'disableRefresh',
                text: 'Disable auto refresh',
                iconProps: closeTimer,
                onClick: getInterval.bind(this, 0)
            },
            {
                key: 'refresh10',
                text: 'Refresh every 10s',
                iconProps: requency,
                onClick: getInterval.bind(this, 10)
            },
            {
                key: 'refresh20',
                text: 'Refresh every 20s',
                iconProps: requency,
                onClick: getInterval.bind(this, 20)
            },
            {
                key: 'refresh30',
                text: 'Refresh every 30s',
                iconProps: requency,
                onClick: getInterval.bind(this, 30)
            },

            {
                key: 'refresh60',
                text: 'Refresh every 1min',
                iconProps: requency,
                onClick: getInterval.bind(this, 60)
            }
        ]
    };
    const aboutProps: IContextualMenuProps = {
        items: [
            {
                key: 'feedback',
                text: 'Feedback',
                iconProps: { iconName: 'OfficeChat' },
                onClick: openGithub
            },
            {
                key: 'help',
                text: 'Document',
                iconProps: { iconName: 'TextDocument' },
                onClick: openDocs
            },
            {
                key: 'version',
                text: `Version ${version}`,
                iconProps: { iconName: 'VerifiedBrand' },
                onClick: openGithubNNI
            }
        ]
    };

    return (
        <Stack horizontal className='nav'>
            <React.Fragment>
                <StackItem grow={30} styles={{ root: { minWidth: 300, display: 'flex', verticalAlign: 'center' } }}>
                    <span className='desktop-logo'>{NNILOGO}</span>
                    <span className='left-right-margin'>{OVERVIEWTABS}</span>
                    <span>{DETAILTABS}</span>
                </StackItem>
                <StackItem grow={70} className='navOptions'>
                    <Stack horizontal horizontalAlign='end' tokens={gap15} styles={stackStyle}>
                        {/* refresh button danyi*/}
                        {/* TODO: fix bug */}
                        {/* <CommandBarButton
                                    iconProps={{ iconName: 'sync' }}
                                    text="Refresh"
                                    onClick={this.props.refreshFunction}
                                /> */}
                        <div className='nav-refresh'>
                            <CommandBarButton
                                iconProps={refreshFrequency === '' ? disableUpdates : timeIcon}
                                text={refreshText}
                                menuProps={refreshProps}
                            />
                            <div className='nav-refresh-num'>{refreshFrequency}</div>
                        </div>
                        <CommandBarButton
                            iconProps={{ iconName: 'ShowResults' }}
                            text='Experiment summary'
                            onClick={(): void => setVisibleExperimentPanel(true)}
                        />
                        <CommandBarButton iconProps={infoIconAbout} text='About' menuProps={aboutProps} />
                        <Link to='/experiment' className='experiment'>
                            <div className='expNavTitle'>
                                <span>All experiments</span>
                                {ChevronRightMed}
                            </div>
                        </Link>
                    </Stack>
                </StackItem>
                {visibleExperimentPanel && (
                    <ExperimentSummaryPanel
                        closeExpPanel={(): void => setVisibleExperimentPanel(false)}
                        experimentProfile={EXPERIMENT.profile}
                    />
                )}
            </React.Fragment>
            {/* experiment error model */}
            <ErrorMessage />
        </Stack>
    );
};

export default NavCon;
