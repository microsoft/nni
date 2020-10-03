import * as React from 'react';
import { DetailsRow, IDetailsRowBaseProps } from 'office-ui-fabric-react';
import OpenRow from '../public-child/OpenRow';

interface ExpandableDetailsProps {
    detailsProps: IDetailsRowBaseProps;
    isExpand: boolean;
}

interface ExpandableDetailsState {
    isExpand: boolean;
}

class ExpandableDetails extends React.Component<ExpandableDetailsProps, ExpandableDetailsState> {
    constructor(props: ExpandableDetailsProps) {
        super(props);
        this.state = { isExpand: false };
    }

    render(): React.ReactNode {
        const { detailsProps, isExpand } = this.props;
        // const { isExpand } = this.state;
        return (
            <div>
                <DetailsRow {...detailsProps} />
                {isExpand && <OpenRow trialId={detailsProps.item.id} />}
            </div>
        );
    }
}

export default ExpandableDetails;
