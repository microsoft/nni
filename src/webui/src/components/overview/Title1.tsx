import * as React from 'react';
import { Stack } from 'office-ui-fabric-react';
import '../../static/style/overviewTitle.scss';
interface Title1Props {
    text: string;
    icon?: string;
    fontColor?: string;
}

class Title1 extends React.Component<Title1Props, {}> {
    constructor(props: Title1Props) {
        super(props);
    }

    render(): React.ReactNode {
        const { text, icon, fontColor } = this.props;
        return (
            <Stack horizontal className='panelTitle'>
                <img src={require(`../../static/img/icon/${icon}`)} alt='icon' />
                <span style={{ color: fontColor }}>{text}</span>
            </Stack>
        );
    }
}

export default Title1;
