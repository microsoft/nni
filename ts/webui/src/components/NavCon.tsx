import * as React from 'react';
import axios from 'axios';
import { WEBUIDOC, MANAGER_IP } from '../static/const';
import { Stack, StackItem, CommandBarButton, IContextualMenuProps } from '@fluentui/react';
import { Link } from 'react-router-dom';
import { infoIconAbout, timeIcon, disableUpdates, requency, closeTimer, ChevronRightMed } from './buttons/Icon';
import ExperimentSummaryPanel from './modals/ExperimentSummaryPanel';
import { OVERVIEWTABS, DETAILTABS, NNILOGO } from './stateless-component/NNItabs';
import { EXPERIMENT } from '../static/datamodel';
import { stackTokens, stackStyle } from './NavConst';
import '../static/style/nav/nav.scss';
import '../static/style/icon.scss';

interface NavState {
    version: string;
    menuVisible: boolean;
    navBarVisible: boolean;
    isdisabledFresh: boolean;
    isvisibleExperimentDrawer: boolean;
    refreshText: string;
    refreshFrequency: number | string;
}

interface NavProps {
    changeInterval: (value: number) => void;
    refreshFunction: () => void;
}

class NavCon extends React.Component<NavProps, NavState> {
    constructor(props: NavProps) {
        super(props);
        this.state = {
            version: '',
            menuVisible: false,
            navBarVisible: false,
            isdisabledFresh: false,
            isvisibleExperimentDrawer: false,
            refreshText: 'Auto refresh',
            refreshFrequency: 10
        };
    }

    // to see & download experiment parameters
    showExpcontent = (): void => {
        this.setState({ isvisibleExperimentDrawer: true });
    };

    // close download experiment parameters drawer
    closeExpDrawer = (): void => {
        this.setState({ isvisibleExperimentDrawer: false });
    };

    getNNIversion = (): void => {
        axios(`${MANAGER_IP}/version`, {
            method: 'GET'
        }).then(res => {
            if (res.status === 200) {
                this.setState({ version: res.data });
            }
        });
    };

    openGithub = (): void => {
        const { version } = this.state;
        const feed = `https://github.com/Microsoft/nni/issues/new?labels=${version}`;
        window.open(feed);
    };

    openDocs = (): void => {
        window.open(WEBUIDOC);
    };

    openGithubNNI = (): void => {
        const { version } = this.state;
        const nniLink = `https://github.com/Microsoft/nni/tree/${version}`;
        window.open(nniLink);
    };

    getInterval = (num: number): void => {
        this.props.changeInterval(num); // notice parent component
        this.setState(() => ({
            refreshFrequency: num === 0 ? '' : num,
            refreshText: num === 0 ? 'Disable auto' : 'Auto refresh'
        }));
    };

    componentDidMount(): void {
        this.getNNIversion();
    }

    render(): React.ReactNode {
        const { isvisibleExperimentDrawer, version, refreshText, refreshFrequency } = this.state;
        const aboutProps: IContextualMenuProps = {
            items: [
                {
                    key: 'feedback',
                    text: 'Feedback',
                    iconProps: { iconName: 'OfficeChat' },
                    onClick: this.openGithub
                },
                {
                    key: 'help',
                    text: 'Document',
                    iconProps: { iconName: 'TextDocument' },
                    onClick: this.openDocs
                },
                {
                    key: 'version',
                    text: `Version ${version}`,
                    iconProps: { iconName: 'VerifiedBrand' },
                    onClick: this.openGithubNNI
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
                        <Stack horizontal horizontalAlign='end' tokens={stackTokens} styles={stackStyle}>
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
                                    menuProps={this.refreshProps}
                                />
                                <div className='nav-refresh-num'>{refreshFrequency}</div>
                            </div>
                            <CommandBarButton
                                iconProps={{ iconName: 'ShowResults' }}
                                text='Experiment summary'
                                onClick={this.showExpcontent}
                            />
                            <CommandBarButton iconProps={infoIconAbout} text='About' menuProps={aboutProps} />
                            <Link to='/experiment' className='experiment'>
                                <div className='expNavTitle'>
                                    <span>All experiment</span>
                                    {ChevronRightMed}
                                </div>
                            </Link>
                        </Stack>
                    </StackItem>
                    {isvisibleExperimentDrawer && (
                        <ExperimentSummaryPanel
                            closeExpDrawer={this.closeExpDrawer}
                            experimentProfile={EXPERIMENT.profile}
                        />
                    )}
                </React.Fragment>
            </Stack>
        );
    }

    private refreshProps: IContextualMenuProps = {
        items: [
            {
                key: 'disableRefresh',
                text: 'Disable auto refresh',
                iconProps: closeTimer,
                onClick: this.getInterval.bind(this, 0)
            },
            {
                key: 'refresh10',
                text: 'Refresh every 10s',
                iconProps: requency,
                onClick: this.getInterval.bind(this, 10)
            },
            {
                key: 'refresh20',
                text: 'Refresh every 20s',
                iconProps: requency,
                onClick: this.getInterval.bind(this, 20)
            },
            {
                key: 'refresh30',
                text: 'Refresh every 30s',
                iconProps: requency,
                onClick: this.getInterval.bind(this, 30)
            },

            {
                key: 'refresh60',
                text: 'Refresh every 1min',
                iconProps: requency,
                onClick: this.getInterval.bind(this, 60)
            }
        ]
    };
}

export default NavCon;
